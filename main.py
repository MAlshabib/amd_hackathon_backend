from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from enum import Enum
import os

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

# Enums
class TransactionStatus(str, Enum):
    pending = "pending"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    refunded = "refunded"
    scheduled = "scheduled"

class TransactionType(str, Enum):
    deposit = "deposit"
    withdrawal = "withdrawal"
    transfer = "transfer"
    fee = "fee"
    refund = "refund"

# Member Model
class Member(BaseModel):
    name: str
    email: str | None = None

# Association Model
class Association(BaseModel):
    name: str
    required_amount: float
    goal: str

# AssociationMember Link Model
class AssociationMember(BaseModel):
    member_id: int
    association_id: int

# Transaction Model
class Transaction(BaseModel):
    member_id: int
    association_id: int | None = None
    type: TransactionType = TransactionType.deposit
    amount: float
    status: TransactionStatus = TransactionStatus.completed
    category: str | None = None
    note: str | None = None

@app.post("/members")
def create_member(member: Member):
    result = supabase.table("members").insert(member.dict()).execute()
    return {"message": "Member created", "data": result.data}

@app.get("/members")
def get_members():
    result = supabase.table("members").select("*").order("created_at", desc=True).execute()
    return {"members": result.data}

@app.post("/associations")
def create_association(association: Association):
    result = supabase.table("associations").insert(association.dict()).execute()
    return {"message": "Association created", "data": result.data}

@app.get("/associations")
def get_associations():
    result = supabase.table("associations").select("*").order("created_at", desc=True).execute()
    return {"associations": result.data}

@app.post("/associations/add-member")
def add_member_to_association(link: AssociationMember):
    result = supabase.table("association_members").insert(link.dict()).execute()
    return {"message": "Member added to association", "data": result.data}

@app.get("/associations/{association_id}/members")
def get_association_members(association_id: int):
    result = supabase.table("association_members").select("*, members(*)").eq("association_id", association_id).execute()
    return {"members": result.data}

@app.post("/transactions")
def create_transaction(txn: Transaction):
    txn_data = txn.dict()

    # Rule 1: Block withdrawals from associations
    if txn_data.get("association_id") and txn_data["type"] == "withdrawal":
        raise HTTPException(status_code=400, detail="Cannot withdraw from an association")

    # Rule 2: Auto-assign category if needed
    if txn_data.get("association_id"):
        txn_data["category"] = txn_data.get("category") or "association-deposit"
    else:
        txn_data["category"] = txn_data.get("category") or "general"

    # Rule 3: Amount must be positive
    if txn_data["amount"] <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than zero")

    # Insert to Supabase
    result = supabase.table("transactions").insert(txn_data).execute()
    return {"message": "Transaction created", "data": result.data}

@app.get("/members/{member_id}/associations/{association_id}/paid")
def get_paid_amount(member_id: int, association_id: int):
    result = supabase.table("transactions") \
        .select("amount") \
        .eq("member_id", member_id) \
        .eq("association_id", association_id) \
        .eq("type", "deposit") \
        .eq("status", "completed") \
        .execute()

    total_paid = sum(txn["amount"] for txn in result.data)
    return {"member_id": member_id, "association_id": association_id, "total_paid": total_paid}

@app.get("/members/{member_id}/transactions")
def get_personal_transactions(member_id: int):
    result = supabase.table("transactions") \
        .select("*") \
        .eq("member_id", member_id) \
        .is_("association_id", None) \
        .order("created_at", desc=True) \
        .execute()

    return {"transactions": result.data}

@app.get("/members/{member_id}/summary")
def get_transaction_summary(member_id: int, source: str = "personal"):
    query = supabase.table("transactions").select("category, amount").eq("member_id", member_id)

    if source == "personal":
        query = query.is_("association_id", None)
    elif source == "association":
        query = query.not_.is_("association_id", None)
    elif source == "all":
        pass
    else:
        raise HTTPException(status_code=400, detail="Invalid source type. Use 'personal', 'association', or 'all'.")

    result = query.execute()

    summary = {}
    for txn in result.data:
        cat = txn["category"]
        summary[cat] = summary.get(cat, 0) + txn["amount"]

    return {"summary": summary}