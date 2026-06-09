import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="OMAI Recruiter AI",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 44px;
    font-weight: 800;
}
.card {
    padding: 22px;
    border-radius: 16px;
    background-color: #111827;
    border: 1px solid #374151;
}
.small-text {
    color: #9CA3AF;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 OMAI Recruiter AI</div>', unsafe_allow_html=True)
st.markdown("### AI-Powered Candidate Discovery & Ranking Dashboard")
st.write("Built for INDIA RUNS Data & AI Challenge")

with st.sidebar:
    st.header("📌 Project")
    st.write("OMAI Recruiter AI ranks candidates using:")
    st.write("✅ Semantic Match")
    st.write("✅ Skill Fit")
    st.write("✅ Experience Fit")
    st.write("✅ Behavioral Signals")
    st.write("✅ Explainable Reasoning")
    st.divider()
    uploaded = st.file_uploader("Upload submission.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    top_candidate = df.sort_values("rank").iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Candidates", len(df))
    c2.metric("Top Score", round(df["score"].max(), 4))
    c3.metric("Average Score", round(df["score"].mean(), 4))
    c4.metric("Top Candidate", top_candidate["candidate_id"])

    st.divider()

    st.subheader("🏆 Top 3 Leaderboard")

    top3 = df.sort_values("rank").head(3)

    l1, l2, l3 = st.columns(3)

    for col, (_, row), medal in zip(
        [l1, l2, l3],
        top3.iterrows(),
        ["🥇", "🥈", "🥉"]
    ):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <h2>{medal} Rank {int(row['rank'])}</h2>
                    <h3>{row['candidate_id']}</h3>
                    <p><b>Score:</b> {round(row['score'], 4)}</p>
                    <p class="small-text">{row['reasoning']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    st.subheader("🔍 Search Candidate")

    search = st.text_input("Enter Candidate ID")

    if search:
        result = df[df["candidate_id"].str.contains(search, case=False, na=False)]

        if not result.empty:
            st.dataframe(result, use_container_width=True)

            selected = result.iloc[0]

            st.info(
                f"""
                Candidate {selected['candidate_id']} is ranked #{int(selected['rank'])}
                with score {round(selected['score'], 4)}.
                
                Reasoning: {selected['reasoning']}
                """
            )
        else:
            st.warning("Candidate not found.")

    st.divider()

    st.subheader("📊 Top 20 Score Distribution")

    top20 = df.sort_values("rank").head(20)

    fig = px.bar(
        top20,
        x="candidate_id",
        y="score",
        text="score",
        title="Top 20 Candidate Scores"
    )

    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Score Trend by Rank")

    fig2 = px.line(
        df.sort_values("rank"),
        x="rank",
        y="score",
        markers=True,
        title="Score Trend Across Top 100 Candidates"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("📋 Ranked Candidate Table")

    st.dataframe(
        df.sort_values("rank"),
        use_container_width=True,
        height=500
    )

    st.download_button(
        "⬇️ Download Ranked CSV",
        df.to_csv(index=False),
        file_name="submission.csv",
        mime="text/csv"
    )

else:
    st.info("Upload your generated submission.csv file from the sidebar to view the dashboard.")