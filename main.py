from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from enum import Enum
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
tadawul_old_prices = pd.read_csv("tadawul_stcks.csv")

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


tadawul_df = pd.read_csv("tadawul_stcks.csv")
tadawul_df = tadawul_df.rename(columns=lambda x: x.strip())

top_symbols = (
    tadawul_df.groupby("symbol")["volume_traded"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index
)

tadawul_df.columns = tadawul_df.columns.str.strip()
tadawul_df['date'] = pd.to_datetime(tadawul_df['date'])
tadawul_df = tadawul_df.sort_values('date')

def predict_stock_with_trend(symbol: int, amount_of_money: float, price_today: float, days_to_invest: int):
    # Filter and prepare data for the specific stock
    stock_df = tadawul_df[tadawul_df['symbol'] == symbol].copy()
    stock_name = stock_df['trading_name'].iloc[0] if not stock_df.empty else "Unknown Stock"
    # stock_df = df[df['symbol']].copy()
    stock_df = stock_df.sort_values('date')
    stock_df['days'] = (stock_df['date'] - stock_df['date'].min()).dt.days

    # Drop missing values
    stock_df = stock_df.dropna(subset=['open', 'high', 'low', 'volume_traded', 'close'])

    if stock_df.shape[0] < 30:
        raise ValueError("Not enough data to train the model for this stock.")

    # Define features and target
    features = ['days', 'open', 'high', 'low', 'volume_traded']
    X = stock_df[features]
    y = stock_df['close']

    # Train/test split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Train the model
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Calculate future input using trend (avg of last 5 days)
    last_n = stock_df.tail(5)
    avg_open = last_n['open'].mean()
    avg_high = last_n['high'].mean()
    avg_low = last_n['low'].mean()
    avg_volume = last_n['volume_traded'].mean()
    future_day = stock_df['days'].max() + days_to_invest

    future_input = [[future_day, avg_open, avg_high, avg_low, avg_volume]]
    predicted_price = model.predict(future_input)[0]

    # Investment calculation
    shares = amount_of_money / price_today
    expected_value = shares * predicted_price
    profit = expected_value - amount_of_money
    roi_percent = (profit / amount_of_money) * 100

    # Accuracy
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return {
        "symbol": symbol,
        "price_today": round(price_today, 2),
        "predicted_price": round(predicted_price, 2),
        "expected_profit": round(profit, 2),
        "expected_roi_percent": round(roi_percent, 2),
        "model_r2_score": round(r2, 4),
        "stock_name": stock_name
    }

@app.get("/stocks/best_stocks")
# Use only these symbols to find top 3 investments
def top_3_filtered_stocks_to_invest(amount_of_money: float, days_to_invest: int):
    results = []
    for symbol in top_symbols:
        try:
            stock_df = tadawul_df[tadawul_df['symbol'] == symbol].copy()
            stock_df = stock_df.sort_values('date')
            price_today = stock_df['close'].iloc[-1]

            result = predict_stock_with_trend(
                symbol=symbol,
                amount_of_money=amount_of_money,
                price_today=price_today,
                days_to_invest=days_to_invest
            )
            results.append(result)
        except Exception:
            continue

    # Sort and return top 3
    top_stocks = sorted(results, key=lambda x: x['expected_profit'], reverse=True)[:3]
    return top_stocks

class InvestmentAccount(BaseModel):
    name: str
    member_id: int

@app.post("/members/create_investment_account")
def create_investment_account(inv_account: InvestmentAccount):
    try:
        req = inv_account.dict()
        print("Request:", req)

        # Check if member exists
        member = supabase.table("members").select("*").eq("id", req["member_id"]).execute()
        if not member.data:
            raise HTTPException(status_code=404, detail="Member not found")

        member = member.data[0]
        account_number_base = member.get("account_number", "ACCT")

        account_data = {
            "account_number": f"{account_number_base}-{req['member_id']:05d}",
            "balance": 0.0,
            "name": req['name']
        }

        result = supabase.table("investment_account").insert(account_data).execute()

        if result.error:
            print("Supabase Error:", result.error)
            raise HTTPException(status_code=500, detail=result.error["message"])

        return {"message": "Investment account created", "data": result.data}

    except Exception as e:
        print("💥 Internal Server Error:", e)
        raise HTTPException(status_code=500, detail=str(e))




@app.get('/members/{member_id}/advanced_summary')
def advanced_summary(member_id: int):
    member_data = supabase.table("members").select("*").eq("id", member_id).execute()
    if not member_data.data:
        raise HTTPException(status_code=404, detail="Member not found")
    member = member_data.data[0]
    transactions = supabase.table("transactions").select("*").eq("member_id", member_id).execute()
    if not transactions.data:
        raise HTTPException(status_code=404, detail="No transactions found for this member")
    total_transactions = len(transactions.data)
    total_amount = sum(txn['amount'] for txn in transactions.data)
    categories = {}
    for txn in transactions.data:
        category = txn['category'] or 'uncategorized'
        if category not in categories:
            categories[category] = 0
        categories[category] += txn['amount']
    summary = {
        "member": member,
        "total_transactions": total_transactions,
        "total_amount": total_amount,
        "categories": categories
    }
    return summary

