import streamlit as st
import recommender as rec   # loads recommender_data.pkl automatically

st.set_page_config(page_title="TGEAPCET College Recommender", layout="wide")
st.title("TGEAPCET College Recommender")
st.caption(f"Based on {rec.latest_year} final-phase closing ranks. "
           "This is a guide, not a guarantee: next year's cutoffs can move a lot.")

latest = rec.latest

col1, col2, col3 = st.columns(3)
rank = col1.number_input("Your rank", min_value=1, max_value=300000, value=14000, step=100)
caste = col2.selectbox("Category", sorted(latest["CASTE"].unique()))
gender = col3.selectbox("Gender", ["B", "G"],
                        format_func=lambda g: "Boy" if g == "B" else "Girl")

branches = st.multiselect("Branches (leave empty for all)", sorted(latest["BRANCH"].unique()))
districts = st.multiselect("Districts (leave empty for all)", sorted(latest["DIST"].dropna().unique()))

shift = st.slider(
    "Expected change in cutoffs vs last year (log scale)",
    min_value=-0.30, max_value=0.30, value=0.0, step=0.05,
    help="0 = same as last year. Negative = cutoffs get smaller. "
         "Yearly changes have ranged from about -0.2 to +0.2 and cannot be predicted.")

show_unlikely = st.checkbox("Also show 'Unlikely' options", value=False)

if st.button("Find colleges"):
    res = rec.recommend(rank, caste, gender,
                        branches=branches or None,
                        districts=districts or None,
                        shift=shift, show_unlikely=show_unlikely)
    if res.empty:
        st.warning("No options found. Try more branches or districts.")
    else:
        counts = res["BUCKET"].value_counts()
        st.write("  |  ".join(f"**{b}**: {counts.get(b, 0)}"
                              for b in ["Safe", "Target", "Reach", "Open (likely available)"]))
        for bucket in ["Safe", "Target", "Reach", "Open (likely available)", "Unlikely"]:
            part = res[res["BUCKET"] == bucket]
            if len(part):
                with st.expander(f"{bucket} ({len(part)})", expanded=(bucket in ["Safe", "Target"])):
                    st.dataframe(part.drop(columns="BUCKET"), use_container_width=True)

st.markdown("---")
st.caption("Safe/Target/Reach come from a backtest on 2019-2025: students ranked at 0.6x of a seat's "
           "previous cutoff were admitted at least 87% of the time in the worst year. "
           "Ranks exclude special categories (PH, NCC, sports, CAP, minority). SC uses the SC_III "
           "series for 2025, a known limitation.")
