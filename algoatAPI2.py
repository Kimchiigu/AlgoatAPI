from datetime import datetime
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch
from rake_nltk import Rake
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uuid
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore

# Firebase Admin SDK setup
cred = credentials.Certificate("firebase-service.json")  # Replace with your Firebase service account key file
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load BERT model and tokenizer for text embedding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load Sentence-BERT model for context analysis
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize RAKE for keyword extraction
rake = Rake()

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Answer(BaseModel):
    player: str
    username: str
    answer: str

class StartGameRequest(BaseModel):
    room_id: str
    participants: List[str]
    category: str
    num_questions: int  # Add num_questions field
    answer_time: int  # Add answer_time field

class QuestionResponse(BaseModel):
    question: str
    phaseTime: str

class ScoreResponse(BaseModel):
    player: str
    username: str
    score: float

class JudgementResponse(BaseModel):
    question: str
    Answers: List[ScoreResponse]
    winner: str

class LeaderboardEntry(BaseModel):
    player: str
    score: int  # Updated to expect an integer

class LeaderboardResponse(BaseModel):
    leaderboard: List[LeaderboardEntry]

class QuestionRequest(BaseModel):
    question: str
    context: str

# Load the dataset
def load_dataset(base_path):
    data = []
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(category_path, filename)
                    with open(file_path, 'r') as file:
                        content = file.read().strip()
                        if content:
                            data.append({
                                'category': category,
                                'question': filename.replace('.txt', ''),
                                'text': content,
                                'context': content
                            })
    df = pd.DataFrame(data)
    df['text_embedding'] = df['text'].apply(encode_text)
    df['context_embedding'] = df['context'].apply(encode_context)
    return df

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()  # Convert to list

def encode_context(text):
    return sbert_model.encode(text).tolist()  # Convert to list

def extract_keywords(text):
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()
    return " ".join(keywords)

def get_best_score(user_input, category_df):
    # Extract context from user input
    user_context = extract_keywords(user_input)

    # Encode user input
    user_text_embedding = encode_text(user_input)
    user_context_embedding = encode_context(user_context)

    # Calculate similarity scores for text
    category_df = category_df.copy()  # Avoiding SettingWithCopyWarning
    category_df.loc[:, 'text_similarity'] = category_df['text_embedding'].apply(lambda x: cosine_similarity([x], [user_text_embedding]).item())

    # Calculate similarity scores for context
    category_df.loc[:, 'context_similarity'] = category_df['context_embedding'].apply(lambda x: cosine_similarity([x], [user_context_embedding]).item())

    # Combine scores
    category_df.loc[:, 'final_score'] = (category_df['text_similarity'] + category_df['context_similarity']) / 2

    # Get the best match
    best_match = category_df.loc[category_df['final_score'].idxmax()]

    return best_match['text'], best_match['final_score']

# Initialize the dataset
base_path = 'Dataset'  # Replace with your actual dataset path
df = load_dataset(base_path)

@app.post("/start_game")
def start_game(request: StartGameRequest):
    session_id = str(uuid.uuid4())
    print("Incoming request:", request.model_dump())
    # Filter the dataset based on the specified category
    filtered_df = df[df['category'] == request.category]
    if filtered_df.empty:
        raise HTTPException(status_code=400, detail="No questions available for the specified category")
    
    print("errorr")
    # Select the specified number of questions
    questions = filtered_df.sample(n=request.num_questions).to_dict(orient="records")
    print("soal 2", [{ "question": q["question"], "category": q["category"] } for q in questions])
    participants = [{"player": p, "score": 0} for p in request.participants]
    
    
    db.collection("Games").document(session_id).set({
        "room_id": request.room_id,
        "questions": [{ "question": q["question"], "category": q["category"] } for q in questions],  # Only save necessary info
        "current_question_index": 0,
        "scores": {},
        "is_playing": True,
        "phase": "question",  # Initial phase
        "phase_start_time": datetime.now().isoformat(),
        "participants": participants,
        "answer_time": request.answer_time,  # Store answer time
        "num_questions": request.num_questions  # Store number of questions
    })
    # Save participants in a sub-collection
    for participant in participants:
        db.collection("Games").document(session_id).collection("Participants").document(participant["player"]).set(participant)
    
    return {"session_id": session_id}


@app.get("/get_question/{session_id}", response_model=QuestionResponse)
def get_question(session_id: str):
    game_doc = db.collection("Games").document(session_id).get()
    if not game_doc.exists:
        raise HTTPException(status_code=404, detail="Session not found")

    game_data = game_doc.to_dict()
    current_question_index = game_data["current_question_index"]
    questions = game_data["questions"]

    if current_question_index >= len(questions):
        raise HTTPException(status_code=404, detail="No more questions available")

    question = questions[current_question_index]["question"]
    current_time = datetime.now().isoformat()
    return QuestionResponse(question=question, phaseTime=current_time)

@app.post("/submit_answer/{session_id}")
def submit_answer(session_id: str, answer: Answer):
    game_doc = db.collection("Games").document(session_id).get()
    if not game_doc.exists:
        raise HTTPException(status_code=404, detail="Session not found")

    answer_dict = answer.model_dump()

    # Store the answer in the Answers sub-collection
    try:
        db.collection("Games").document(session_id).collection("Answers")
        db.collection("Games").document(session_id).collection("Answers").add(answer_dict)
        print(f"Answer saved successfully: {answer_dict} in {session_id}")
    except Exception as e:
        print(f"Error saving answer: {e}")
        raise HTTPException(status_code=500, detail="Error saving answer")

    # Update the count of submitted Answers
    game_data = game_doc.to_dict()
    current_count = game_data.get("submitted_answers", 0) + 1
    db.collection("Games").document(session_id).update({"submitted_answers": current_count})
    
@app.post("/check_game_state/{session_id}")
def check_game_state(session_id: str):
    game_doc = db.collection("Games").document(session_id).get()
    if not game_doc.exists:
        raise HTTPException(status_code=404, detail="Session not found")

    game_data = game_doc.to_dict()
    if not game_data["is_playing"]:
        return {"status": "ended"}

    current_time = datetime.now()
    phase_start_time = datetime.fromisoformat(game_data["phase_start_time"])

    if game_data["phase"] == "question":
        if (current_time - phase_start_time).seconds >= 10:
            db.collection("Games").document(session_id).update({
                "phase": "answer",
                "phase_start_time": current_time.isoformat()
            })
            return {"status": "answer", "question": game_data["questions"][game_data["current_question_index"]]["question"], "phaseTime": game_data["phase_start_time"]}

    elif game_data["phase"] == "answer":
        if (current_time - phase_start_time).seconds >= (game_data["answer_time"]*60):
            calculate_scores(session_id)
            db.collection("Games").document(session_id).update({
                "phase": "judging",
                "phase_start_time": current_time.isoformat()
            })
            return {"status": "judging"}
        else:
            return {"status": "answer", "phaseTime": game_data["phase_start_time"]}

    elif game_data["phase"] == "judging":
        if (current_time - phase_start_time).seconds >= 5:
            next_question_index = game_data["current_question_index"] + 1
            db.collection("Games").document(session_id).update({
                "phase": "leaderboard",
                "phase_start_time": current_time.isoformat()
            })
            return {"status": "leaderboard"}
    
    elif game_data["phase"] == "leaderboard":
        if (current_time - phase_start_time).seconds >= 5:  # Show leaderboard for 10 seconds
            next_question_index = game_data["current_question_index"]
            if next_question_index < len(game_data["questions"]):
                db.collection("Games").document(session_id).update({
                    "phase": "question",
                    "phase_start_time": current_time.isoformat()
                })
                return {"status": "question"}
            else:
                db.collection("Games").document(session_id).update({
                    "is_playing": False
                })
                return {"status": "ended"}

    return {"status": game_data["phase"]}


@app.post("/calculate_scores/{session_id}", response_model=JudgementResponse)
def calculate_scores(session_id: str):
    game_doc = db.collection("Games").document(session_id).get()
    if not game_doc.exists:
        raise HTTPException(status_code=404, detail="Session not found")

    game_data = game_doc.to_dict()
    current_question_index = game_data["current_question_index"]
    questions = game_data["questions"]

    if current_question_index == -1:
        raise HTTPException(status_code=400, detail="No question has been fetched yet")

    current_question = questions[current_question_index]

    # Retrieve all Answers for the current round
    answers = []
    answers_docs = db.collection("Games").document(session_id).collection("Answers").stream()
    for doc in answers_docs:
        doc_data = doc.to_dict()
        print(f"Fetched answer document: {doc_data}")  # Add logging
        try:
            answer = Answer(player=doc_data["player"], username=doc_data["username"], answer=doc_data["answer"])
            answers.append(answer)
        except KeyError as e:
            print(f"KeyError: {e} in document: {doc_data}")
        except Exception as e:
            print(f"Unexpected error: {e} in document: {doc_data}")

    # Use local dataset for scoring
    category_df = df[df['category'] == current_question['category']].copy()

    scores = []
    for answer in answers:
        best_text, best_score = get_best_score(answer.answer, category_df)
        scores.append(ScoreResponse(player=answer.player, username=answer.username, score=best_score))

    winner = ''
    if scores:
        winner_score = max(scores, key=lambda x: x.score)
        winner = winner_score.username
        print(winner_score)
        # Increment the winner's score
        print(winner_score.player)
        participant_doc_ref = db.collection("Games").document(session_id).collection("Participants").document(winner_score.username)
        participant_doc = participant_doc_ref.get()
        if participant_doc.exists:
            participant_data = participant_doc.to_dict()
            participant_data["score"] += 1
            participant_doc_ref.update({"score": participant_data["score"]})
            print(f"Updated score for {winner_score.username}: {participant_data['score']}")

    # Update the game session in Firestore
    db.collection("Games").document(session_id).update({
        "current_question_index": current_question_index + 1,
        "scores": game_data["scores"],
        "is_playing": True,
        "winner": winner
    })

    # Clear the Answers sub-collection for the next round
    answers_ref = db.collection("Games").document(session_id).collection("Answers")
    for doc in answers_ref.stream():
        doc.reference.delete()

    return JudgementResponse(
        question=current_question["question"],
        Answers=scores,
        winner=winner
    )

@app.get("/get_leaderboard/{session_id}", response_model=LeaderboardResponse)
def get_leaderboard(session_id: str):
    participants_ref = db.collection("Games").document(session_id).collection("Participants")
    participants_docs = participants_ref.stream()
    leaderboard = [{"player": doc.to_dict()["player"], "score": doc.to_dict()["score"]} for doc in participants_docs]

    leaderboard.sort(key=lambda x: x["score"], reverse=True)  # Sort by score in descending order
    return LeaderboardResponse(leaderboard=leaderboard)

@app.post("/end_game/{session_id}")
def end_game(session_id: str):
    game_doc = db.collection("Games").document(session_id).get()
    if not game_doc.exists:
        raise HTTPException(status_code=404, detail="Session not found")

    game_data = game_doc.to_dict()
    room_id = game_data["room_id"]

    # Update the room document to remove sessionId and set isPlay to false
    room_doc_ref = db.collection("Rooms").document(room_id)
    room_doc_ref.update({
        "sessionId": firestore.DELETE_FIELD,
        "isPlay": False
    })

    # Delete the game session from Firestore
    db.collection("Games").document(session_id).delete()

    return {"message": "Game Ended"}

from pydantic import BaseModel
from transformers import pipeline, BertTokenizerFast, AlbertForQuestionAnswering

# Load the fine-tuned model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained('Wikidepia/indobert-lite-squad')
model = AlbertForQuestionAnswering.from_pretrained('Wikidepia/indobert-lite-squad')

# Create the QA pipeline
qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer
)

@app.post("/answer/")
async def get_answer(question_request: QuestionRequest):
    result = qa_pipeline({
        'context': question_request.context,
        'question': question_request.question
    })
    return {"answer": result['answer']}

# To run the API, use the command: uvicorn script_name:app --reload
# Replace 'script_name' with the name of your script file
# uvicorn algoatAPI2:app --reload
