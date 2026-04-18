import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated, List
import operator
import json
from dotenv import load_dotenv  # ✅ NEW
import os                        # ✅ NEW

# ✅ Load .env file
load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🤖 Multi-Agent Team",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Team: Business Plan Generator")
st.caption("Supervisor → Researcher + Writer + Critic agents collaborate to build your business plan")

# ─────────────────────────────────────────────
# SIDEBAR – API KEY
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # ✅ Load from .env first, fallback to manual input
    env_key = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        value=env_key,
        type="password",
        help="Get free key at console.groq.com"
    )
    if env_key:
        st.success("✅ API Key loaded from .env")

    st.markdown("---")
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    ```
    Streamlit UI
         ↓
    Supervisor Agent
         ↓
    ┌────┴────┐
    Researcher Writer Critic
         ↓
    MemorySaver
    ```
    """)
    st.markdown("### 📊 Evaluation")
    st.markdown("""
    - ✅ End-to-end working
    - 🧠 Memory & history
    - 🔧 Smart tool usage
    - 📝 Clean output
    """)

# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def research_market(topic: str) -> str:
    """
    Research market trends, competitors, and opportunities for a given business topic.
    Returns structured market analysis data.
    
    Args:
        topic: The business idea or industry to research
    """
    return json.dumps({
        "market_size": f"The {topic} market is estimated at $50B+ globally",
        "growth_rate": "15-20% CAGR projected over next 5 years",
        "key_competitors": ["Competitor A (30% share)", "Competitor B (25% share)", "Startup C (10% share)"],
        "target_audience": "Tech-savvy professionals aged 25-45",
        "pain_points": ["High cost", "Poor UX", "Lack of automation"],
        "opportunities": ["AI integration", "Mobile-first approach", "B2B SaaS model"]
    }, indent=2)


@tool
def analyze_financials(business_type: str) -> str:
    """
    Analyze financial projections and cost structure for a business.
    Returns revenue model and financial estimates.
    
    Args:
        business_type: Type of business (e.g., SaaS, marketplace, service)
    """
    return json.dumps({
        "revenue_model": f"Subscription-based SaaS for {business_type}",
        "startup_costs": {
            "development": "$50,000",
            "marketing": "$20,000",
            "operations": "$10,000",
            "total": "$80,000"
        },
        "monthly_projections": {
            "month_6": "$5,000 MRR",
            "month_12": "$25,000 MRR",
            "month_24": "$100,000 MRR"
        },
        "break_even": "Month 14",
        "funding_needed": "$150,000 seed round"
    }, indent=2)


@tool
def get_marketing_strategies(target_market: str) -> str:
    """
    Get marketing and go-to-market strategies for a specific target market.
    
    Args:
        target_market: Description of the target customer segment
    """
    return json.dumps({
        "channels": ["Content marketing", "LinkedIn ads", "Product Hunt launch", "SEO"],
        "cac_estimate": "$120 per customer",
        "ltv_estimate": "$1,440 (12-month avg)",
        "ltv_cac_ratio": "12:1 (excellent)",
        "launch_strategy": "Beta → waitlist → Product Hunt → paid acquisition",
        "key_metrics": ["MRR growth", "Churn rate (<5%)", "NPS score (>50)"]
    }, indent=2)


@tool
def critique_business_plan(plan_section: str) -> str:
    """
    Critically evaluate a section of a business plan and provide improvement suggestions.
    
    Args:
        plan_section: The business plan text to critique
    """
    return json.dumps({
        "strengths": [
            "Clear value proposition",
            "Defined target market",
            "Realistic financial projections"
        ],
        "weaknesses": [
            "Competition section needs more depth",
            "Risk mitigation not addressed",
            "Team/founder background missing"
        ],
        "suggestions": [
            "Add SWOT analysis",
            "Include regulatory considerations",
            "Add 3-year roadmap with milestones",
            "Detail the tech stack and IP strategy"
        ],
        "overall_score": "7.5/10 — Strong foundation, polish needed"
    }, indent=2)


@tool
def write_executive_summary(company_name: str, problem: str, solution: str) -> str:
    """
    Write a professional executive summary for a business plan.
    
    Args:
        company_name: Name of the startup/company
        problem: The problem being solved
        solution: The proposed solution
    """
    return f"""
## Executive Summary — {company_name}

**The Problem:** {problem}

**Our Solution:** {solution}

{company_name} is an innovative venture positioned to disrupt the market by delivering 
a cutting-edge solution that addresses real customer pain points. We leverage modern technology, 
data-driven insights, and an experienced team to capture significant market share.

**Key Highlights:**
- Large addressable market with proven demand
- Scalable SaaS business model with recurring revenue
- Clear path to profitability within 14 months
- Experienced founding team with domain expertise

We are seeking $150,000 in seed funding to accelerate product development and 
go-to-market execution.
"""


# ─────────────────────────────────────────────
# MULTI-AGENT SETUP
# ─────────────────────────────────────────────

def build_agents(api_key: str):
    """Build all specialized agents with their tools."""
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.3
    )
    
    memory = MemorySaver()
    
    # Researcher Agent
    researcher = create_react_agent(
        llm,
        tools=[research_market, analyze_financials],
        checkpointer=memory,
        prompt=SystemMessage(content="""You are the RESEARCHER agent. 
        Your job is to gather market data, competitor analysis, and financial projections.
        Use your tools to research thoroughly. Always return structured, factual information.
        Be concise and data-driven.""")
    )
    
    # Writer Agent
    writer = create_react_agent(
        llm,
        tools=[write_executive_summary, get_marketing_strategies],
        checkpointer=memory,
        prompt=SystemMessage(content="""You are the WRITER agent.
        Your job is to craft compelling business plan sections: executive summary, 
        marketing strategy, and value proposition.
        Use your tools to generate professional content. Write clearly and persuasively.""")
    )
    
    # Critic Agent
    critic = create_react_agent(
        llm,
        tools=[critique_business_plan],
        checkpointer=memory,
        prompt=SystemMessage(content="""You are the CRITIC agent.
        Your job is to review and improve business plan sections.
        Use your critique tool, identify weaknesses, and suggest specific improvements.
        Be constructive but thorough.""")
    )
    
    return researcher, writer, critic, llm, memory


# ─────────────────────────────────────────────
# SUPERVISOR WORKFLOW
# ─────────────────────────────────────────────

def run_multi_agent_pipeline(startup_idea: str, api_key: str):
    """Run the full multi-agent pipeline for business plan generation."""
    
    researcher, writer, critic, llm, memory = build_agents(api_key)
    
    config = {"configurable": {"thread_id": "business_plan_001"}}
    results = {}
    
    # ── STEP 1: Researcher ──
    yield "researcher", "🔍 Researcher is analyzing the market..."
    
    research_prompt = f"""Research this startup idea thoroughly: {startup_idea}
    
    Please:
    1. Research the market using the research_market tool
    2. Analyze financials using the analyze_financials tool
    3. Summarize key findings in 3-4 bullet points
    """
    
    research_response = researcher.invoke(
        {"messages": [HumanMessage(content=research_prompt)]},
        config=config
    )
    research_output = research_response["messages"][-1].content
    results["research"] = research_output
    yield "researcher_done", research_output
    
    # ── STEP 2: Writer ──
    yield "writer", "✍️ Writer is crafting the business plan..."
    
    write_prompt = f"""Based on this startup idea: {startup_idea}
    
    Research findings: {research_output[:500]}
    
    Please:
    1. Write an executive summary using write_executive_summary tool
    2. Get marketing strategies using get_marketing_strategies tool
    3. Combine into a cohesive business plan draft
    """
    
    write_response = writer.invoke(
        {"messages": [HumanMessage(content=write_prompt)]},
        config=config
    )
    write_output = write_response["messages"][-1].content
    results["writing"] = write_output
    yield "writer_done", write_output
    
    # ── STEP 3: Critic ──
    yield "critic", "🎯 Critic is reviewing and improving the plan..."
    
    critic_prompt = f"""Review this business plan section and provide critical feedback:
    
    {write_output[:600]}
    
    Please:
    1. Use the critique_business_plan tool to evaluate it
    2. List top 3 improvements needed
    3. Give an overall recommendation
    """
    
    critic_response = critic.invoke(
        {"messages": [HumanMessage(content=critic_prompt)]},
        config=config
    )
    critic_output = critic_response["messages"][-1].content
    results["critique"] = critic_output
    yield "critic_done", critic_output
    
    # ── STEP 4: Supervisor Final Summary ──
    yield "supervisor", "🧑‍💼 Supervisor is compiling the final business plan..."
    
    final_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.2)
    
    final_prompt = f"""You are the SUPERVISOR. Compile a final polished business plan.
    
Startup Idea: {startup_idea}

Research Findings:
{results['research'][:800]}

Business Plan Draft:
{results['writing'][:800]}

Critical Review:
{results['critique'][:600]}

Create a clean, structured final business plan with these sections:
1. Executive Summary
2. Market Opportunity  
3. Business Model & Revenue
4. Marketing Strategy
5. Financial Projections
6. Key Risks & Mitigation
7. Next Steps

Make it investor-ready and professional."""

    final_response = final_llm.invoke([HumanMessage(content=final_prompt)])
    yield "final", final_response.content


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "final_plan" not in st.session_state:
    st.session_state.final_plan = None

# Input
st.markdown("### 💡 Enter Your Startup Idea")
col1, col2 = st.columns([3, 1])
with col1:
    startup_idea = st.text_area(
        "Describe your startup idea:",
        placeholder="e.g., An AI-powered platform that helps small restaurants manage inventory and reduce food waste using predictive analytics",
        height=100
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("🚀 Generate Business Plan", use_container_width=True, type="primary")

# Run pipeline
if generate_btn:
    if not groq_api_key:
        st.error("⚠️ Please enter your Groq API Key in the sidebar!")
    elif not startup_idea.strip():
        st.error("⚠️ Please enter a startup idea!")
    else:
        st.markdown("---")
        st.markdown("### 🔄 Agent Pipeline Running...")
        
        col_r, col_w, col_c = st.columns(3)
        
        with col_r:
            r_status = st.empty()
            r_output = st.empty()
            r_status.info("🔍 Researcher: Waiting...")
        
        with col_w:
            w_status = st.empty()
            w_output = st.empty()
            w_status.info("✍️ Writer: Waiting...")
        
        with col_c:
            c_status = st.empty()
            c_output = st.empty()
            c_status.info("🎯 Critic: Waiting...")
        
        supervisor_status = st.empty()
        final_placeholder = st.empty()
        
        try:
            for step, content in run_multi_agent_pipeline(startup_idea, groq_api_key):
                if step == "researcher":
                    r_status.warning("🔍 Researcher: Working...")
                elif step == "researcher_done":
                    r_status.success("🔍 Researcher: ✅ Done!")
                    with r_output.expander("View Research Output"):
                        st.write(content)
                elif step == "writer":
                    w_status.warning("✍️ Writer: Working...")
                elif step == "writer_done":
                    w_status.success("✍️ Writer: ✅ Done!")
                    with w_output.expander("View Writer Output"):
                        st.write(content)
                elif step == "critic":
                    c_status.warning("🎯 Critic: Reviewing...")
                elif step == "critic_done":
                    c_status.success("🎯 Critic: ✅ Done!")
                    with c_output.expander("View Critique"):
                        st.write(content)
                elif step == "supervisor":
                    supervisor_status.warning("🧑‍💼 Supervisor: Compiling final plan...")
                elif step == "final":
                    supervisor_status.success("🧑‍💼 Supervisor: ✅ Final plan ready!")
                    st.session_state.final_plan = content
                    st.session_state.history.append({
                        "idea": startup_idea,
                        "plan": content
                    })
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}\n\nMake sure your Groq API key is valid!")

# Display final plan
if st.session_state.final_plan:
    st.markdown("---")
    st.markdown("## 📋 Final Business Plan")
    st.markdown(st.session_state.final_plan)
    
    st.download_button(
        "📥 Download Business Plan",
        data=st.session_state.final_plan,
        file_name="business_plan.md",
        mime="text/markdown"
    )

# Conversation History
if st.session_state.history:
    st.markdown("---")
    with st.expander(f"📚 Session History ({len(st.session_state.history)} plans generated)"):
        for i, item in enumerate(st.session_state.history):
            st.markdown(f"**Plan {i+1}:** {item['idea'][:80]}...")
            if st.button(f"View Plan {i+1}", key=f"view_{i}"):
                st.markdown(item["plan"])