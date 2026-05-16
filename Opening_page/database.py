from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi import FastAPI, File, UploadFile
from chainlit.utils import mount_chainlit

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pydantic import BaseModel
from jose import jwt, JWTError
from passlib.hash import bcrypt
from sqlalchemy import Column, Integer, String, Text, create_engine, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from openai import AsyncOpenAI
from fpdf import FPDF
from dotenv import load_dotenv
import os

load_dotenv("deploy.env")

API_URL = os.getenv("API_URL")
FASTAPI_URL = os.getenv("FASTAPI_URL")
LLM_URL = os.getenv("LLM_URL").rstrip('/')

class DoctorVisitPDF(FPDF):
    def header(self):
        self.image('./public/logo_dark.png', 10, 8, 20)
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, "DOCTOR'S VISIT SUMMARY", ln=True, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(0)
        self.cell(0, 10, title, ln=True)
        self.set_font('Arial', '', 11)
        self.set_text_color(50)

    def multi_line_list(self, items):
        for item in items:
            self.cell(5)
            self.multi_cell(0, 8, f"- {item}")

            
SECRET_KEY = "C@odmUUt5H4$%UPob*zQDXcl=:q_fjXwFwlE-9cuXA?LjofbpbrHwsKA7SE5Fh6A"
ALGORITHM = "HS256"

openai_client = AsyncOpenAI(
    base_url=f"{LLM_URL}/v1",
    api_key=""
)

Base = declarative_base()
engine = create_engine("sqlite:///db_data/chat.db")
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    content_question = Column(Text)
    content_answer = Column(Text)
    question_id = Column(Integer)
    chronology = Column(Integer)

Base.metadata.create_all(bind=engine)
app = FastAPI(title = 'MenoChat App', version = '1.0.0')
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str


app.mount("/templates", StaticFiles(directory="templates"), name="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve specific file on route
@app.get("/")
def get_home_page():
    return FileResponse(os.path.join("templates", "dashboard.html"))
    
@app.get("/register-page")
def get_register_page():
    return FileResponse(os.path.join("templates", "register.html"))

# @app.get("/login-page")
# def get_login_page():
#     return FileResponse(os.path.join("templates", "login.html"))

@app.post("/register", response_model=Token)
def register(user: UserCreate):
    db = SessionLocal()
    if db.query(User).filter_by(username=user.username).first():
        raise HTTPException(status_code=400, detail="Username taken")
    hashed_pw = bcrypt.hash(user.password)
    db_user = User(username=user.username, password=hashed_pw)
    db.add(db_user)
    db.commit()
    token = jwt.encode({"sub": user.username}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/login", response_model=Token)
def login(user: UserCreate):
    db = SessionLocal()
    db_user = db.query(User).filter_by(username=user.username).first()
    if not db_user or not bcrypt.verify(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = jwt.encode({"sub": user.username}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}


def get_current_user(token: str = Header(...)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/profile")
def profile_page():
    return FileResponse(os.path.join("templates", "profile.html"))
@app.get("/me")
def get_me(current_user: str = Depends(get_current_user)):
    return {"username": current_user}



class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str
    
class AuthToken(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str or None = None



# app = FastAPI(title = 'MenoChat App', version = '1.0.0')
# bge_model = BGE(system_prompt = None)




@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        with open(file.filename, 'wb') as f:
            while contents := file.file.read(1024 * 1024):
                f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()

    # return ChatResponse(response = 'returned_'+str(file.filename))

    return FileResponse(
        path='UPGP_U001_Q1.mp3',
        filename=f"returned_{file.filename}",
        media_type=file.content_type
    )

#Chat Response
@app.post("/chat", response_model = ChatResponse)
async def chat(request: ChatRequest):
    #AI implementation
    
    # return_text = bge_model.chat(request.prompt)
    return_text = 'this is the critical text we are trying to get.'
    
    return ChatResponse(response = str(return_text))
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncpg
from typing import Optional, List, Dict
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database class (same as yours, adapted)
import aiosqlite
import re
import os
from typing import Optional, Dict, List

class Run_database():
    def __init__(self):
        self.database_path = 'database.db'
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self):
        if not self._db:
            self._db = await aiosqlite.connect(self.database_path)

    async def execute_query(self, query: str, params=None) -> List[Dict]:
        if not self._db:
            await self.connect()
        try:
            sqlite_query = re.sub(r'\$\d+', '?', query)

            if params is not None:
                if isinstance(params, dict):
                    param_values = list(params.values())
                else:
                    param_values = list(params)

                expanded_params = []
                new_query_parts = []
                i = 0
                parts = sqlite_query.split('?')
                for j, part in enumerate(parts[:-1]):
                    val = param_values[i] if i < len(param_values) else None
                    if isinstance(val, list):
                        placeholders = ', '.join(['?'] * len(val))
                        new_query_parts.append(part.rstrip())
                        new_query_parts[-1] = re.sub(r'\(\s*$', '', new_query_parts[-1])
                        new_query_parts.append(f'({placeholders})')
                        expanded_params.extend(val)
                    else:
                        new_query_parts.append(part)
                        new_query_parts.append('?')
                        expanded_params.append(val)
                    i += 1
                new_query_parts.append(parts[-1])
                sqlite_query = ''.join(new_query_parts)
                param_values = expanded_params
            else:
                param_values = []

            async with self._db.execute(sqlite_query, param_values) as cursor:
                rows = await cursor.fetchall()
                if cursor.description is None:
                    return []
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            print(f"DB error: {e}")
            return []


db = Run_database()
templates = Jinja2Templates(directory="templates")


@app.get("/admin_index", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/identifiers")
async def get_identifiers():
    query = 'SELECT DISTINCT identifier FROM "User" ORDER BY identifier;'
    result = await db.execute_query(query)
    return [r['identifier'] for r in result]


@app.get("/steps/{identifier}")
async def get_user_steps(identifier: str):
    # Step 1: Get thread IDs for the user
    thread_query = """
    SELECT "Thread".id
    FROM "Thread"
    JOIN "User" ON "Thread"."userId" = "User".id
    WHERE "User".identifier = ?;
    """
    thread_rows = await db.execute_query(thread_query, [identifier])
    thread_ids = [row["id"] for row in thread_rows]

    if not thread_ids:
        return []

    # Step 2: Fetch steps of type 'user_message' for those threads
    step_query = """
    SELECT "Step".*
    FROM "Step"
    JOIN "Thread" ON "Step"."threadId" = "Thread".id
    WHERE "Thread".id IN (?) AND "Step".type = 'user_message';
    """
    steps = await db.execute_query(step_query, [thread_ids])

    step_query_assist = """
    SELECT "Step".*
    FROM "Step"
    JOIN "Thread" ON "Step"."threadId" = "Thread".id
    WHERE "Thread".id IN (?) AND "Step".type = 'assistant_message';
    """
    steps_assist = await db.execute_query(step_query_assist, [thread_ids])

    step_string = ''
    step_list = []
    step_string_assist = ''
    step_list_assist = []
    message_history = []

    for step in steps:
        step_list.append(step['output'])
        step_string += step['output'] + '\n'

    for step in steps_assist:
        step_list_assist.append(step['output'])
        step_string_assist += step['output'] + '\n'

    message_history.append({
        "role": "system",
        "content": """Classify each input as either a:
- "knowledge_ask" (seeking information), or
- "symptom_query" (describing symptoms).
Only for menstrual or menopausal questions or asks. Not anything else. Ignore anything else.
For symptom queries:
- Extract symptoms with "name", "mentions", and "severity" (mild, moderate, severe).
For knowledge asks:
- Extract key "concepts" with how many times they're mentioned.
Return output in pure english. Do not add emojis or text that cannot be coded in a pdf.
Return a structured output string easy to understand."""
    })

    message_history.append({"role": "user", "content": step_string})

    response = await openai_client.chat.completions.create(
        model="meno",
        messages=message_history,
        temperature=0.2
    )
    summary = response.choices[0].message.content

    url = f"/pdfs/doctors_visit_summary_{identifier}.pdf"
    filename = f'AI_doctor_generated_{identifier}.pdf'
    patient_name = identifier
    visit_date = "2025-07-16"
    problems = [
        "Persistent headache for 3 weeks",
        "Occasional dizziness and fatigue",
        "Mild lower back pain"
    ]
    recommendations = [
        "Stay hydrated and reduce screen time.",
        "Begin light stretching or yoga every morning.",
        "Schedule a follow-up in 2 weeks."
    ]

    pdf = DoctorVisitPDF()
    pdf.add_page()
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, f"Date: {visit_date}", ln=True)
    pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.ln(5)

    pdf.section_title("Problems Reported:")
    pdf.multi_line_list(problems)
    pdf.ln(5)

    pdf.section_title("Doctor's Recommendations:")
    pdf.multi_line_list(recommendations)
    pdf.ln(5)

    pdf.section_title("Tests Ordered:")
    pdf.multi_cell(0, 10, "_______________________________\n\n_______________________________\n\n_______________________________")

    pdf.output(f"./static/pdfs/doctors_visit_summary_.pdf")

    return {
        "step_list": step_list,
        "summary": summary,
        "url": url,
        "filename": filename
    }


@app.get("/pdfs/{filename}")
async def serve_pdf(filename: str):
    base_dir = os.path.abspath("static/pdfs")
    file_path = os.path.join(base_dir, filename)

    if not file_path.startswith(base_dir) or not os.path.isfile(file_path):
        return {"error": "File not found"}

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="AI_doctor_generated.pdf"'
        }
    )
