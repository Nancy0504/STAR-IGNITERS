import gradio as gr
import json
import os
from datetime import datetime, timedelta
import speech_recognition as sr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Any
import calendar
import requests

# Initialize Gemini API
GEMINI_API_KEY = "AIzaSyAYHskNyBjAT3x49uVKc6op268p1mOh2ME"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def query_gemini_api(prompt: str, context: str = "") -> str:
    """Query Google Gemini API for financial advice"""
    
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("⚠️ Gemini API key not configured, falling back to rule-based advisor")
        return get_rule_based_advice(prompt, context)
    
    try:
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': GEMINI_API_KEY
        }
        
        system_prompt = f"""You are a professional financial advisor AI. Based on the user's financial data, provide personalized, actionable advice.

Financial Context:
{context}

Provide helpful, practical financial advice. Be concise but comprehensive. Include specific recommendations when possible. Keep responses under 300 words and format with clear sections using bullet points where appropriate."""
        
        full_prompt = f"{system_prompt}\n\nUser question: {prompt}"
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": full_prompt
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(GEMINI_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                return f"🤖 **Gemini Financial Advisor:**\n\n{content}"
        else:
            print(f"Gemini API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
    
    # Fall back to rule-based advice
    return f"💡 **Financial Advisor:**\n\n{get_rule_based_advice(prompt, context)}"

print("🤖 Loading Financial Tracker with Gemini AI...")
print(f"🔑 Gemini API Status: {'Configured' if GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE' else 'Not configured - using fallback'}")

class FinancialTracker:
    def __init__(self):
        self.data = {
            "expenses": [],
            "savings": [],
            "income": [],
            "budgets": [],
            "tax_info": []
        }
        self.load_data()
    
    def load_data(self):
        """Load existing data from JSON file"""
        try:
            if os.path.exists("financial_data.json"):
                with open("financial_data.json", "r") as f:
                    self.data = json.load(f)
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def save_data(self):
        """Save data to JSON file"""
        try:
            with open("financial_data.json", "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def add_expense(self, amount: float, category: str, description: str = "") -> str:
        """Add new expense"""
        expense = {
            "id": len(self.data["expenses"]) + 1,
            "amount": amount,
            "category": category,
            "description": description,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "expense"
        }
        self.data["expenses"].append(expense)
        self.save_data()
        return f"✅ Added expense: ${amount:.2f} for {category}"
    
    def add_savings(self, amount: float, goal: str, description: str = "") -> str:
        """Add savings entry"""
        savings = {
            "id": len(self.data["savings"]) + 1,
            "amount": amount,
            "goal": goal,
            "description": description,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "savings"
        }
        self.data["savings"].append(savings)
        self.save_data()
        return f"💰 Added savings: ${amount:.2f} towards {goal}"
    
    def add_income(self, amount: float, source: str, description: str = "") -> str:
        """Add income entry"""
        income = {
            "id": len(self.data["income"]) + 1,
            "amount": amount,
            "source": source,
            "description": description,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "income"
        }
        self.data["income"].append(income)
        self.save_data()
        return f"📈 Added income: ${amount:.2f} from {source}"
    
    def set_budget(self, category: str, amount: float, period: str = "monthly") -> str:
        """Set budget for a category"""
        budget = {
            "id": len(self.data["budgets"]) + 1,
            "category": category,
            "amount": amount,
            "period": period,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Remove existing budget for same category
        self.data["budgets"] = [b for b in self.data["budgets"] if b["category"] != category]
        self.data["budgets"].append(budget)
        self.save_data()
        return f"📊 Set {period} budget: ${amount:.2f} for {category}"
    
    def calculate_tax_estimate(self, annual_income: float, filing_status: str = "single") -> Dict:
        """Calculate basic tax estimate (US tax brackets 2024)"""
        tax_brackets = {
            "single": [
                (11000, 0.10),
                (44725, 0.12),
                (95375, 0.22),
                (182050, 0.24),
                (231250, 0.32),
                (578125, 0.35),
                (float('inf'), 0.37)
            ],
            "married": [
                (22000, 0.10),
                (89450, 0.12),
                (190750, 0.22),
                (364200, 0.24),
                (462500, 0.32),
                (693750, 0.35),
                (float('inf'), 0.37)
            ]
        }
        
        brackets = tax_brackets.get(filing_status, tax_brackets["single"])
        tax_owed = 0
        remaining_income = annual_income
        
        prev_bracket = 0
        for bracket, rate in brackets:
            taxable_in_bracket = min(remaining_income, bracket - prev_bracket)
            if taxable_in_bracket <= 0:
                break
            tax_owed += taxable_in_bracket * rate
            remaining_income -= taxable_in_bracket
            prev_bracket = bracket
        
        tax_info = {
            "annual_income": annual_income,
            "estimated_tax": tax_owed,
            "after_tax_income": annual_income - tax_owed,
            "effective_rate": (tax_owed / annual_income) * 100 if annual_income > 0 else 0,
            "filing_status": filing_status,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.data["tax_info"].append(tax_info)
        self.save_data()
        
        return tax_info
    
    def get_summary(self) -> Dict:
        """Get financial summary"""
        total_expenses = sum([exp["amount"] for exp in self.data["expenses"]])
        total_savings = sum([sav["amount"] for sav in self.data["savings"]])
        total_income = sum([inc["amount"] for inc in self.data["income"]])
        
        return {
            "total_income": total_income,
            "total_expenses": total_expenses,
            "total_savings": total_savings,
            "net_worth": total_income - total_expenses + total_savings,
            "savings_rate": (total_savings / total_income * 100) if total_income > 0 else 0
        }
    
    def get_budget_status(self) -> Dict:
        """Get budget vs actual spending comparison"""
        budget_status = {}
        current_month = datetime.now().strftime("%Y-%m")
        
        # Get current month expenses
        monthly_expenses = {}
        for expense in self.data["expenses"]:
            if expense["date"].startswith(current_month):
                category = expense["category"]
                monthly_expenses[category] = monthly_expenses.get(category, 0) + expense["amount"]
        
        # Compare with budgets
        for budget in self.data["budgets"]:
            category = budget["category"]
            budgeted = budget["amount"]
            spent = monthly_expenses.get(category, 0)
            remaining = budgeted - spent
            
            budget_status[category] = {
                "budgeted": budgeted,
                "spent": spent,
                "remaining": remaining,
                "percentage": (spent / budgeted * 100) if budgeted > 0 else 0
            }
        
        return budget_status
    
    def get_transaction_history(self, limit: int = 50) -> List[Dict]:
        """Get recent transaction history"""
        all_transactions = []
        
        # Add all transactions with proper formatting
        for expense in self.data["expenses"]:
            all_transactions.append({
                **expense,
                "amount": -expense["amount"],  # Negative for expenses
                "category": expense["category"]
            })
        
        for income in self.data["income"]:
            all_transactions.append({
                **income,
                "category": income["source"]
            })
        
        for savings in self.data["savings"]:
            all_transactions.append({
                **savings,
                "category": savings["goal"]
            })
        
        # Sort by date (newest first)
        all_transactions.sort(key=lambda x: x["date"], reverse=True)
        
        return all_transactions[:limit]

# Initialize tracker
tracker = FinancialTracker()

def get_rule_based_advice(prompt: str, context: str) -> str:
    """Rule-based financial advice when AI model is not available"""
    prompt_lower = prompt.lower()
    
    # Parse financial context
    lines = context.strip().split('\n')
    financial_data = {}
    for line in lines:
        try:
            if 'Total Income:' in line:
                financial_data['income'] = float(line.split('$')[1].replace(',', ''))
            elif 'Total Expenses:' in line:
                financial_data['expenses'] = float(line.split('$')[1].replace(',', ''))
            elif 'Savings Rate:' in line:
                financial_data['savings_rate'] = float(line.split(':')[1].replace('%', ''))
            elif 'Net Worth:' in line:
                financial_data['net_worth'] = float(line.split('$')[1].replace(',', ''))
        except:
            continue
    
    # Generate advice based on keywords and financial status
    if any(word in prompt_lower for word in ['budget', 'budgeting', 'spend']):
        if financial_data.get('savings_rate', 0) < 10:
            return """🏦 **Budgeting Advice:**
            
Based on your data, your savings rate could be improved. Here are some budgeting tips:

• **50/30/20 Rule**: Allocate 50% for needs, 30% for wants, 20% for savings
• **Track every expense** for at least one month to identify spending patterns  
• **Cut one unnecessary subscription** or service this month
• **Meal prep** to reduce food expenses (often 20-30% savings)
• **Review your largest expense categories** and find 10% cuts in each

💡 *Tip: Start with small changes - they compound over time!*"""
        else:
            return """✅ **Great Budgeting Progress!**
            
Your savings rate shows good financial discipline. Here's how to optimize further:

• **Zero-based budgeting**: Assign every dollar a purpose
• **Automate savings** to remove temptation
• **Review and optimize** recurring subscriptions quarterly
• **Consider the envelope method** for discretionary spending
• **Set specific financial goals** with timelines

🎯 *Next step: Consider increasing your emergency fund to 6 months of expenses*"""
    
    elif any(word in prompt_lower for word in ['save', 'saving', 'emergency']):
        return """💰 **Savings Strategy:**
        
Priority order for your savings:

1. **Emergency Fund**: 3-6 months of expenses (start with $1,000)
2. **High-interest debt**: Pay off credit cards (usually 15-25% interest)
3. **Employer 401k match**: Free money (typically 3-6% of salary)
4. **Roth IRA**: $6,500/year limit (2024), tax-free growth
5. **Additional investments**: Index funds, real estate, etc.

🔄 **Automate it**: Set up automatic transfers on payday
📈 **Start small**: Even $25/week builds the habit
🎯 **Use the 1% rule**: Increase savings by 1% every 3 months"""
    
    elif any(word in prompt_lower for word in ['invest', 'investment', 'retirement']):
        return """📊 **Investment Guidance:**
        
**For Beginners:**
• **Start with index funds** (S&P 500, Total Market)
• **Dollar-cost averaging**: Invest same amount regularly
• **Keep costs low**: Look for expense ratios under 0.2%
• **Diversify**: Don't put all eggs in one basket

**Timeline-based approach:**
• **0-5 years**: High-yield savings, CDs, money market
• **5-10 years**: Balanced portfolio (60% stocks, 40% bonds)  
• **10+ years**: Growth-focused (80-90% stocks)

⚠️ *Never invest money you need within 5 years*
💡 *Consider target-date funds for hands-off investing*"""
    
    elif any(word in prompt_lower for word in ['debt', 'loan', 'credit']):
        return """💳 **Debt Management:**
        
**Two main strategies:**

🔥 **Debt Avalanche** (mathematically optimal):
• Pay minimums on all debts
• Put extra money toward highest interest rate debt
• Save more money on interest long-term

⚡ **Debt Snowball** (psychological wins):
• Pay minimums on all debts  
• Put extra money toward smallest balance
• Build momentum with quick wins

**Credit improvement tips:**
• Pay on time (35% of credit score)
• Keep utilization under 30% (30% of credit score)
• Don't close old accounts (15% of credit score)
• Check credit report annually for errors"""
    
    elif any(word in prompt_lower for word in ['tax', 'taxes', 'deduction']):
        return """🧮 **Tax Optimization:**
        
**Common deductions to consider:**
• **Standard vs. Itemized**: Compare what saves more
• **Retirement contributions**: 401k, IRA reduce taxable income
• **Health Savings Account (HSA)**: Triple tax advantage
• **Charitable donations**: Keep receipts
• **Home office expenses**: If you work from home

**Tax-efficient strategies:**
• **Max out employer 401k match** first
• **Contribute to Roth IRA** if in lower tax bracket
• **Harvest tax losses** in investment accounts
• **Time large purchases** for maximum deductions

📋 *Consider consulting a tax professional for complex situations*"""
    
    else:
        # General advice based on financial status
        if financial_data.get('net_worth', 0) < 0:
            return """🚨 **Focus on Financial Stability:**
            
Your net worth suggests focusing on basics first:

1. **Create a basic budget** and stick to it
2. **Build a small emergency fund** ($500-1000)
3. **Pay minimum on all debts** to avoid penalties  
4. **Look for ways to increase income** (side gig, skills training)
5. **Cut non-essential expenses** temporarily

💪 *Remember: Small steps lead to big changes over time!*
📱 *Consider using budgeting apps to track spending automatically*"""
        
        elif financial_data.get('savings_rate', 0) > 20:
            return """🌟 **Advanced Financial Planning:**
            
You're doing great! Here's how to optimize further:

• **Tax-loss harvesting** in investment accounts
• **Consider real estate** if you have stable income
• **Explore side income streams** to accelerate goals
• **Review insurance coverage** (life, disability, umbrella)
• **Estate planning** (will, beneficiaries, power of attorney)
• **Consider working with a fee-only financial advisor**

🎯 *Focus on optimizing rather than major changes*"""
        
        else:
            return """📈 **Balanced Financial Health:**
            
You're on a good track! Here are next steps:

• **Automate your finances** (bills, savings, investments)
• **Increase emergency fund** to 6 months expenses
• **Review and optimize** investment portfolio annually
• **Consider additional income streams**
• **Set specific financial goals** with deadlines
• **Learn about tax-advantaged accounts** (HSA, IRA, 401k)

💡 *The key is consistency - small improvements compound over time!*"""

def chatbot_response(message: str, history: List) -> Tuple[str, List]:
    """Handle chatbot interactions"""
    if not message.strip():
        return "", history
    
    summary = tracker.get_summary()
    budget_status = tracker.get_budget_status()
    
    context = f"""
    Current Financial Status:
    - Total Income: ${summary['total_income']:.2f}
    - Total Expenses: ${summary['total_expenses']:.2f}
    - Total Savings: ${summary['total_savings']:.2f}
    - Net Worth: ${summary['net_worth']:.2f}
    - Savings Rate: {summary['savings_rate']:.1f}%
    
    Budget Status: {budget_status}
    """
    
    response = query_gemini_api(message, context)
    history.append([message, response])
    
    return "", history

def process_voice_input(audio_file) -> str:
    """Process voice input and convert to text"""
    if audio_file is None:
        return "No audio file provided"
    
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return f"Voice recognized: {text}"
    except Exception as e:
        return f"Error processing voice: {str(e)}"

# Interface Functions
def add_expense_interface(amount: float, category: str, description: str = "") -> Tuple[str, Any]:
    if amount <= 0:
        return "Please enter a valid amount greater than 0", create_expense_chart()
    result = tracker.add_expense(amount, category, description)
    return result, create_expense_chart()

def add_income_interface(amount: float, source: str, description: str = "") -> Tuple[str, Any]:
    if amount <= 0:
        return "Please enter a valid amount greater than 0", create_financial_overview()
    result = tracker.add_income(amount, source, description)
    return result, create_financial_overview()

def add_savings_interface(amount: float, goal: str, description: str = "") -> Tuple[str, Any]:
    if amount <= 0:
        return "Please enter a valid amount greater than 0", create_savings_chart()
    result = tracker.add_savings(amount, goal, description)
    return result, create_savings_chart()

def set_budget_interface(category: str, amount: float, period: str = "monthly") -> Tuple[str, Any]:
    if amount <= 0:
        return "Please enter a valid amount greater than 0", create_budget_chart()
    result = tracker.set_budget(category, amount, period)
    return result, create_budget_chart()

def calculate_tax_interface(annual_income: float, filing_status: str) -> str:
    if annual_income <= 0:
        return "Please enter a valid annual income"
    
    tax_info = tracker.calculate_tax_estimate(annual_income, filing_status)
    
    return f"""
📊 Tax Estimate for ${annual_income:,.2f} ({filing_status}):

💰 Estimated Tax Owed: ${tax_info['estimated_tax']:,.2f}
💵 After-Tax Income: ${tax_info['after_tax_income']:,.2f}
📈 Effective Tax Rate: {tax_info['effective_rate']:.2f}%
📅 Calculated on: {tax_info['date']}
"""

# Chart Functions
def create_expense_chart():
    if not tracker.data["expenses"]:
        fig = go.Figure()
        fig.add_annotation(text="No expense data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df = pd.DataFrame(tracker.data["expenses"])
    category_totals = df.groupby("category")["amount"].sum()
    
    fig = px.pie(values=category_totals.values, names=category_totals.index, 
                title="Expense Breakdown", hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_savings_chart():
    if not tracker.data["savings"]:
        fig = go.Figure()
        fig.add_annotation(text="No savings data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df = pd.DataFrame(tracker.data["savings"])
    goal_totals = df.groupby("goal")["amount"].sum()
    
    fig = px.bar(x=goal_totals.index, y=goal_totals.values, 
                title="Savings Progress by Goal", color=goal_totals.values,
                color_continuous_scale="Blues")
    fig.update_layout(xaxis_title="Goals", yaxis_title="Amount ($)")
    return fig

def create_financial_overview():
    summary = tracker.get_summary()
    
    fig = go.Figure(data=[
        go.Bar(name='Income', x=['Financial Overview'], y=[summary["total_income"]], marker_color='green'),
        go.Bar(name='Expenses', x=['Financial Overview'], y=[summary["total_expenses"]], marker_color='red'),
        go.Bar(name='Savings', x=['Financial Overview'], y=[summary["total_savings"]], marker_color='blue')
    ])
    
    fig.update_layout(title="Financial Overview", yaxis_title="Amount ($)", barmode='group')
    return fig

def create_budget_chart():
    budget_status = tracker.get_budget_status()
    
    if not budget_status:
        fig = go.Figure()
        fig.add_annotation(text="No budget data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    categories = list(budget_status.keys())
    budgeted = [budget_status[cat]["budgeted"] for cat in categories]
    spent = [budget_status[cat]["spent"] for cat in categories]
    
    fig = go.Figure(data=[
        go.Bar(name='Budgeted', x=categories, y=budgeted, marker_color='lightblue'),
        go.Bar(name='Spent', x=categories, y=spent, marker_color='darkblue')
    ])
    
    fig.update_layout(title="Budget vs Actual Spending", yaxis_title="Amount ($)", barmode='group')
    return fig

def create_transaction_history_table():
    transactions = tracker.get_transaction_history(20)
    
    if not transactions:
        return pd.DataFrame({"Message": ["No transactions found"]})
    
    df_data = []
    for t in transactions:
        df_data.append({
            "Date": t["date"].split()[0],
            "Type": t["type"].title(),
            "Category": t["category"],
            "Amount": f"${abs(t['amount']):.2f}",
            "Description": t.get("description", "")
        })
    
    return pd.DataFrame(df_data)

def get_financial_summary() -> str:
    summary = tracker.get_summary()
    budget_status = tracker.get_budget_status()
    
    budget_text = ""
    if budget_status:
        budget_text = "\n📊 Budget Status:\n"
        for category, status in budget_status.items():
            status_emoji = "✅" if status["remaining"] >= 0 else "⚠️"
            budget_text += f"{status_emoji} {category}: ${status['spent']:.2f}/${status['budgeted']:.2f} ({status['percentage']:.0f}%)\n"
    
    return f"""
📈 Financial Dashboard

💰 Total Income: ${summary['total_income']:,.2f}
💸 Total Expenses: ${summary['total_expenses']:,.2f}
🏦 Total Savings: ${summary['total_savings']:,.2f}
📊 Net Worth: ${summary['net_worth']:,.2f}
💯 Savings Rate: {summary['savings_rate']:.1f}%
{budget_text}
"""

def generate_financial_report() -> str:
    summary = tracker.get_summary()
    transactions = tracker.get_transaction_history()
    
    # Monthly analysis
    current_month = datetime.now().strftime("%Y-%m")
    monthly_expenses = sum([abs(t["amount"]) for t in transactions 
                          if t["type"] == "expense" and t["date"].startswith(current_month)])
    monthly_income = sum([t["amount"] for t in transactions 
                         if t["type"] == "income" and t["date"].startswith(current_month)])
    
    report = f"""
📋 Financial Report - {datetime.now().strftime('%B %Y')}

Overall Summary:
• Total Income: ${summary['total_income']:,.2f}
• Total Expenses: ${summary['total_expenses']:,.2f}
• Total Savings: ${summary['total_savings']:,.2f}
• Net Worth: ${summary['net_worth']:,.2f}

This Month:
• Monthly Income: ${monthly_income:,.2f}
• Monthly Expenses: ${monthly_expenses:,.2f}
• Monthly Net: ${monthly_income - monthly_expenses:,.2f}

Insights:
• Savings Rate: {summary['savings_rate']:.1f}%
• Total Transactions: {len(transactions)}

Recommendations:
• {"Excellent savings rate!" if summary['savings_rate'] > 20 else "Consider increasing your savings rate to at least 20%"}
• {"Great job maintaining positive cash flow!" if summary['net_worth'] > 0 else "Focus on reducing expenses or increasing income"}
"""
    return report

# Create Gradio Interface
with gr.Blocks(title="Personal Financial Tracker", theme=gr.themes.Soft()) as app:
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #2E8B57;">💰 Smart Financial Tracker</h1>
        <p style="color: #666;">AI-powered personal finance management with Google Gemini</p>
    </div>
    """)
    
    with gr.Tabs():
        # Dashboard Tab
        with gr.Tab("📈 Dashboard", id="dashboard"):
            with gr.Row():
                with gr.Column(scale=2):
                    summary_display = gr.Textbox(
                        label="Financial Summary", 
                        value=get_financial_summary(), 
                        interactive=False, 
                        lines=12
                    )
                    refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="primary")
                
                with gr.Column(scale=2):
                    overview_chart = gr.Plot(label="Financial Overview")
        
        # AI Advisor Tab
        with gr.Tab("🤖 AI Financial Advisor", id="advisor"):
            gr.Markdown("### Get personalized financial advice powered by Google Gemini")
            gr.Markdown("💡 Ask about budgeting, saving strategies, investment advice, debt management, and more!")
            
            # Status indicator
            ai_status = gr.HTML(
                value="<div style='color: green; font-weight: bold;'>🟢 Gemini AI: Online and Ready</div>" 
                if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE" 
                else "<div style='color: orange; font-weight: bold;'>🟡 AI Advisor: Configure API key for full features</div>"
            )
            
            chatbot = gr.Chatbot(height=400, show_label=False, bubble_full_width=False)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask: 'How can I improve my savings rate?' or 'Should I pay off debt first?'", 
                    show_label=False,
                    scale=4
                )
                clear_chat = gr.Button("Clear", scale=1)
        
        # Expense Tracker Tab
        with gr.Tab("💸 Expenses", id="expenses"):
            with gr.Row():
                with gr.Column():
                    exp_amount = gr.Number(label="Amount ($)", minimum=0, value=0)
                    exp_category = gr.Dropdown(
                        choices=["Food & Dining", "Transportation", "Housing", "Entertainment", 
                               "Healthcare", "Shopping", "Utilities", "Insurance", "Education", "Other"],
                        label="Category",
                        value="Food & Dining"
                    )
                    exp_description = gr.Textbox(label="Description (optional)", placeholder="Coffee with friends")
                    exp_submit = gr.Button("➕ Add Expense", variant="primary")
                    exp_result = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column():
                    expense_chart = gr.Plot(label="Expense Breakdown")
        
        # Income Tab
        with gr.Tab("📈 Income", id="income"):
            with gr.Row():
                with gr.Column():
                    inc_amount = gr.Number(label="Amount ($)", minimum=0, value=0)
                    inc_source = gr.Dropdown(
                        choices=["Salary", "Freelance", "Business", "Investment", "Gift", "Other"],
                        label="Source",
                        value="Salary"
                    )
                    inc_description = gr.Textbox(label="Description (optional)")
                    inc_submit = gr.Button("➕ Add Income", variant="primary")
                    inc_result = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column():
                    savings_chart = gr.Plot(label="Savings Progress")
        
        # Savings Tab
        with gr.Tab("🏦 Savings", id="savings"):
            with gr.Row():
                with gr.Column():
                    sav_amount = gr.Number(label="Amount ($)", minimum=0, value=0)
                    sav_goal = gr.Dropdown(
                        choices=["Emergency Fund", "Vacation", "House Down Payment", "Car", "Retirement", "Investment", "Other"],
                        label="Savings Goal",
                        value="Emergency Fund"
                    )
                    sav_description = gr.Textbox(label="Description (optional)")
                    sav_submit = gr.Button("➕ Add to Savings", variant="primary")
                    sav_result = gr.Textbox(label="Status", interactive=False)
        
        # Budget Management Tab
        with gr.Tab("📊 Budget", id="budget"):
            with gr.Row():
                with gr.Column():
                    budget_category = gr.Dropdown(
                        choices=["Food & Dining", "Transportation", "Housing", "Entertainment", 
                               "Healthcare", "Shopping", "Utilities", "Insurance", "Education", "Other"],
                        label="Category",
                        value="Food & Dining"
                    )
                    budget_amount = gr.Number(label="Budget Amount ($)", minimum=0, value=0)
                    budget_period = gr.Dropdown(
                        choices=["monthly", "weekly", "yearly"],
                        label="Period",
                        value="monthly"
                    )
                    budget_submit = gr.Button("💰 Set Budget", variant="primary")
                    budget_result = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column():
                    budget_chart = gr.Plot(label="Budget vs Spending")
        
        # Tax Calculator Tab
        with gr.Tab("🧮 Tax Calculator", id="tax"):
            with gr.Row():
                with gr.Column():
                    tax_income = gr.Number(label="Annual Income ($)", minimum=0, value=0)
                    tax_status = gr.Dropdown(
                        choices=["single", "married"],
                        label="Filing Status",
                        value="single"
                    )
                    tax_submit = gr.Button("📋 Calculate Tax", variant="primary")
                
                with gr.Column():
                    tax_result = gr.Textbox(label="Tax Estimate", interactive=False, lines=10)
        
        # Transaction History Tab
        with gr.Tab("📜 History", id="history"):
            with gr.Row():
                history_refresh = gr.Button("🔄 Refresh History", variant="primary")
            transaction_table = gr.Dataframe(
                value=create_transaction_history_table(),
                label="Recent Transactions",
                interactive=False
            )
        
        # Voice Input Tab
        with gr.Tab("🎤 Voice Input", id="voice"):
            gr.Markdown("### Record financial transactions using voice commands")
            with gr.Row():
                with gr.Column():
                    voice_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Voice")
                    voice_submit = gr.Button("🎯 Process Voice", variant="primary")
                    voice_result = gr.Textbox(label="Voice Recognition Result", interactive=False)
        
        # Analytics & Reports Tab
        with gr.Tab("📊 Analytics", id="analytics"):
            with gr.Row():
                report_btn = gr.Button("📋 Generate Report", variant="primary")
            financial_report = gr.Textbox(
                label="Financial Analysis Report",
                interactive=False,
                lines=20,
                value=generate_financial_report()
            )
    
    # Event Handlers
    msg.submit(chatbot_response, [msg, chatbot], [msg, chatbot])
    clear_chat.click(lambda: ([], ""), outputs=[chatbot, msg])
    
    exp_submit.click(add_expense_interface, [exp_amount, exp_category, exp_description], [exp_result, expense_chart])
    inc_submit.click(add_income_interface, [inc_amount, inc_source, inc_description], [inc_result, overview_chart])
    sav_submit.click(add_savings_interface, [sav_amount, sav_goal, sav_description], [sav_result, savings_chart])
    budget_submit.click(set_budget_interface, [budget_category, budget_amount, budget_period], [budget_result, budget_chart])
    
    tax_submit.click(calculate_tax_interface, [tax_income, tax_status], [tax_result])
    voice_submit.click(process_voice_input, [voice_input], [voice_result])
    
    refresh_btn.click(
        lambda: (get_financial_summary(), create_financial_overview()),
        outputs=[summary_display, overview_chart]
    )
    
    history_refresh.click(
        create_transaction_history_table,
        outputs=[transaction_table]
    )
    
    report_btn.click(
        generate_financial_report,
        outputs=[financial_report]
    )
    
    # Load initial data
    app.load(
        lambda: (
            create_expense_chart(),
            create_savings_chart(),
            create_financial_overview(),
            create_budget_chart(),
            create_transaction_history_table()
        ),
        outputs=[expense_chart, savings_chart, overview_chart, budget_chart, transaction_table]
    )

if __name__ == "__main__":
    print("🚀 Starting Financial Tracker with Gemini AI...")
    print(f"🤖 AI Status: {'Gemini API Ready' if GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE' else 'Rule-based fallback'}")
    
    try:
        # Try to launch with share=True first
        app.launch(share=True, debug=True)
    except Exception as e:
        print(f"⚠️  Share link unavailable: {e}")
        print("📱 Running locally only...")
        # Fallback to local only
        app.launch(share=False, debug=True)