import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# =========================
# CONFIG & FILE PATHS
# =========================

DATA_DIR = Path("sff_data")
FINANCE_FILE = DATA_DIR / "sff_finance.csv"
PROJECT_FILE = DATA_DIR / "sff_projects.csv"
CONTACTS_FILE = DATA_DIR / "sff_contacts.csv"
NOTES_FILE = DATA_DIR / "sff_notes.csv"
GOALS_FILE = DATA_DIR / "sff_goals.csv"

DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="SFF HQ – Foundation OS",
    layout="wide"
)

# =========================
# BRANDING (SFF look & feel)
# =========================

def apply_branding():
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1120px;
            padding-top: 1.5rem;
        }
        body {
            background-color: #f5f7fb;
        }
        .sff-sidebar-header {
            font-weight: 700;
            font-size: 1.1rem;
            margin-bottom: 0.15rem;
        }
        .sff-tagline {
            font-size: 0.85rem;
            color: #6b7280;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_branding()

# =========================
# SIMPLE AUTH
# =========================

# You can change these passwords anytime
USERS = {
    "arav": {"password": "visionary", "role": "admin"},
    "board": {"password": "boardview", "role": "board"},
}

if "user" not in st.session_state:
    st.session_state["user"] = None


def login_view():
    st.title("🔐 SFF HQ Login")
    st.write("Internal dashboard for the Soni Family Foundation.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Log in"):
        if username in USERS and password == USERS[username]["password"]:
            st.session_state["user"] = {
                "name": username,
                "role": USERS[username]["role"],
            }
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")


def ensure_auth():
    if not st.session_state["user"]:
        login_view()
        st.stop()


ensure_auth()
role = st.session_state["user"]["role"]

# Sidebar header + logout + public site link
st.sidebar.markdown(
    "<div class='sff-sidebar-header'>Soni Family Foundation</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<div class='sff-tagline'>Unlocking Opportunity Through Education</div>",
    unsafe_allow_html=True,
)
st.sidebar.write(f"👤 **{st.session_state['user']['name']}** ({role})")
st.sidebar.markdown(
    "[🌐 View public site](https://www.sonifoundation.org) ",
)

if st.sidebar.button("Logout"):
    st.session_state["user"] = None
    st.rerun()

# =========================
# DATA HELPERS
# =========================

@st.cache_data
def load_finance():
    if FINANCE_FILE.exists():
        df = pd.read_csv(FINANCE_FILE, parse_dates=["date"])
        return df
    else:
        return pd.DataFrame(
            columns=["date", "category", "type", "project", "description", "amount"]
        )


def save_finance(df: pd.DataFrame):
    df.to_csv(FINANCE_FILE, index=False)
    load_finance.clear()


@st.cache_data
def load_projects():
    if PROJECT_FILE.exists():
        df = pd.read_csv(PROJECT_FILE)
        return df
    else:
        return pd.DataFrame(
            columns=["project", "category", "status", "budget", "spent", "notes"]
        )


def save_projects(df: pd.DataFrame):
    df.to_csv(PROJECT_FILE, index=False)
    load_projects.clear()


@st.cache_data
def load_contacts():
    if CONTACTS_FILE.exists():
        df = pd.read_csv(CONTACTS_FILE)
        return df
    else:
        return pd.DataFrame(
            columns=["name", "type", "organization", "email", "phone", "notes"]
        )


def save_contacts(df: pd.DataFrame):
    df.to_csv(CONTACTS_FILE, index=False)
    load_contacts.clear()


@st.cache_data
def load_notes():
    if NOTES_FILE.exists():
        df = pd.read_csv(NOTES_FILE, parse_dates=["date"])
        return df
    else:
        return pd.DataFrame(
            columns=[
                "date",
                "meeting_type",
                "title",
                "summary",
                "decisions",
                "attendees",
            ]
        )


def save_notes(df: pd.DataFrame):
    df.to_csv(NOTES_FILE, index=False)
    load_notes.clear()


@st.cache_data
def load_goals():
    if GOALS_FILE.exists():
        df = pd.read_csv(GOALS_FILE, parse_dates=["target_date"])
        return df
    else:
        return pd.DataFrame(
            columns=[
                "goal",
                "owner",
                "area",
                "target_date",
                "status",
                "progress",
                "notes",
            ]
        )


def save_goals(df: pd.DataFrame):
    df.to_csv(GOALS_FILE, index=False)
    load_goals.clear()

# =========================
# KPI & PDF HELPERS
# =========================

def compute_kpis(finance_df: pd.DataFrame):
    if finance_df.empty:
        return {
            "total_income": 0.0,
            "total_expense": 0.0,
            "net_cash": 0.0,
            "program_spend": 0.0,
            "operating_spend": 0.0,
            "program_ratio": 0.0,
        }

    total_income = finance_df.loc[finance_df["type"] == "Income", "amount"].sum()
    total_expense = finance_df.loc[finance_df["type"] == "Expense", "amount"].sum()

    expenses = finance_df[finance_df["type"] == "Expense"]
    program_spend = expenses.loc[expenses["category"] != "Operating", "amount"].sum()
    operating_spend = expenses.loc[expenses["category"] == "Operating", "amount"].sum()

    prog_abs = abs(program_spend)
    op_abs = abs(operating_spend)
    total_spend = prog_abs + op_abs

    program_ratio = (prog_abs / total_spend) if total_spend > 0 else 0.0
    net_cash = total_income + total_expense

    return {
        "total_income": total_income,
        "total_expense": total_expense,
        "net_cash": net_cash,
        "program_spend": prog_abs,
        "operating_spend": op_abs,
        "program_ratio": program_ratio,
    }


def generate_board_report_pdf(finance_df, projects_df) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    kpis = compute_kpis(finance_df)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Soni Family Foundation – Board Report")
    y -= 30

    c.setFont("Helvetica", 12)
    c.drawString(50, y, "1. High-Level Financials")
    y -= 20
    c.drawString(70, y, f"Total Income (YTD): ${kpis['total_income']:,.0f}")
    y -= 15
    c.drawString(70, y, f"Total Spend (YTD): ${kpis['total_expense'] * -1:,.0f}")
    y -= 15
    c.drawString(70, y, f"Net Cash Position: ${kpis['net_cash']:,.0f}")
    y -= 25

    c.drawString(50, y, "2. Program vs Operating")
    y -= 20
    c.drawString(70, y, f"Program Spend: ${kpis['program_spend']:,.0f}")
    y -= 15
    c.drawString(70, y, f"Operating Spend: ${kpis['operating_spend']:,.0f}")
    y -= 15
    c.drawString(70, y, f"Program Ratio: {kpis['program_ratio'] * 100:.1f}%")
    y -= 25

    c.drawString(50, y, "3. Projects Overview")
    y -= 20

    if not projects_df.empty:
        total_projects = len(projects_df)
        active = (projects_df["status"] == "Active").sum()
        planned = (projects_df["status"] == "Planned").sum()
        completed = (projects_df["status"] == "Completed").sum()

        c.drawString(70, y, f"Total projects: {total_projects}")
        y -= 15
        c.drawString(70, y, f"Active: {active}, Planned: {planned}, Completed: {completed}")
        y -= 15
        total_budget = projects_df["budget"].sum()
        total_spent = projects_df["spent"].sum()
        if total_budget > 0:
            util = (total_spent / total_budget) * 100
            c.drawString(70, y, f"Overall budget utilization: {util:.1f}%")
            y -= 20

        c.drawString(70, y, "Top projects by budget:")
        y -= 15
        top = projects_df.sort_values("budget", ascending=False).head(5)
        for _, row in top.iterrows():
            line = (
                f"- {row['project']} ({row['status']}) – "
                f"Budget ${row['budget']:,.0f}, Spent ${row['spent']:,.0f}"
            )
            if y < 80:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 12)
            c.drawString(80, y, line)
            y -= 15
    else:
        c.drawString(70, y, "No projects to display yet.")
        y -= 20

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

# =========================
# LOAD ALL DATA
# =========================

finance_df = load_finance()
projects_df = load_projects()
contacts_df = load_contacts()
notes_df = load_notes()
goals_df = load_goals()
kpis = compute_kpis(finance_df)

# =========================
# SIDEBAR NAVIGATION (ROLE-BASED)
# =========================

st.sidebar.title("Control Panel")

if role == "admin":
    menu_items = [
        "Dashboard",
        "Finance Log",
        "Projects",
        "Scenarios",
        "Contacts",
        "Notes",
        "Goals",
        "Board View",
        "Settings",
    ]
elif role == "board":
    menu_items = [
        "Dashboard",
        "Projects",
        "Contacts",
        "Notes",
        "Goals",
        "Board View",
    ]
else:
    menu_items = ["Dashboard"]

page = st.sidebar.radio("Go to", menu_items)

# =========================
# PAGES
# =========================

# ---- DASHBOARD ----
if page == "Dashboard":
    st.title("📊 SFF HQ – Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Income", f"${kpis['total_income']:,.0f}")
    with col2:
        st.metric("Total Expenses", f"${kpis['total_expense'] * -1:,.0f}")
    with col3:
        st.metric("Net Cash", f"${kpis['net_cash']:,.0f}")

    st.markdown("---")

    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Program vs Operating Spend")
        if kpis["program_spend"] + kpis["operating_spend"] > 0:
            ratio_df = pd.DataFrame(
                {
                    "Category": ["Program", "Operating"],
                    "Amount": [kpis["program_spend"], kpis["operating_spend"]],
                }
            ).set_index("Category")
            st.bar_chart(ratio_df["Amount"])
            st.caption(
                f"Program Ratio: **{kpis['program_ratio'] * 100:.1f}%** "
                f"(target often > 70–80%)"
            )
        else:
            st.info("No expense data yet to compute ratios.")

    with col5:
        if not projects_df.empty:
            st.subheader("Projects Snapshot")
            num_projects = len(projects_df)
            active = (projects_df["status"] == "Active").sum()
            planned = (projects_df["status"] == "Planned").sum()
            completed = (projects_df["status"] == "Completed").sum()

            st.write(f"Total projects: **{num_projects}**")
            st.write(f"Active: **{active}**, Planned: **{planned}**, Completed: **{completed}**")

            total_budget = projects_df["budget"].sum()
            total_spent = projects_df["spent"].sum()
            if total_budget > 0:
                st.write(
                    f"Overall budget utilization: "
                    f"**{(total_spent / total_budget) * 100:.1f}%**"
                )
        else:
            st.info("No projects yet. Add some in the Projects page.")

    st.markdown("---")

    if not finance_df.empty:
        st.subheader("Cash Balance Over Time")
        fin_sorted = finance_df.sort_values("date").copy()
        fin_sorted["cumulative_cash"] = fin_sorted["amount"].cumsum()
        st.line_chart(fin_sorted.set_index("date")["cumulative_cash"])

        st.subheader("Spending by Category (Expenses Only)")
        expense_by_cat = (
            finance_df[finance_df["type"] == "Expense"]
            .groupby("category")["amount"]
            .sum()
            .abs()
            .sort_values(ascending=False)
        )
        if not expense_by_cat.empty:
            st.bar_chart(expense_by_cat)
        else:
            st.info("No expense data yet.")
    else:
        st.info("No finance entries yet. Add some in the Finance Log page.")

# ---- FINANCE LOG ----
elif page == "Finance Log":
    st.title("💸 SFF Finance Log")

    if role != "admin":
        st.info("Only admin can manage the full finance log.")
    else:
        tab_view, tab_add, tab_import = st.tabs(
            ["View / Filter", "Add Entry", "Import from CSV"]
        )

        # VIEW / FILTER
        with tab_view:
            if finance_df.empty:
                st.info("No finance entries found.")
            else:
                st.subheader("All Finance Entries")

                with st.expander("Filters", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        type_filter = st.multiselect(
                            "Type",
                            options=sorted(finance_df["type"].dropna().unique().tolist()),
                            default=sorted(finance_df["type"].dropna().unique().tolist()),
                        )
                    with col2:
                        category_filter = st.multiselect(
                            "Category",
                            options=sorted(
                                finance_df["category"].dropna().unique().tolist()
                            ),
                            default=sorted(
                                finance_df["category"].dropna().unique().tolist()
                            ),
                        )
                    with col3:
                        project_filter = st.multiselect(
                            "Project",
                            options=sorted(
                                finance_df["project"].dropna().unique().tolist()
                            ),
                            default=sorted(
                                finance_df["project"].dropna().unique().tolist()
                            ),
                        )

                filtered = finance_df.copy()
                if type_filter:
                    filtered = filtered[filtered["type"].isin(type_filter)]
                if category_filter:
                    filtered = filtered[filtered["category"].isin(category_filter)]
                if project_filter:
                    filtered = filtered[filtered["project"].isin(project_filter)]

                st.dataframe(filtered.sort_values("date", ascending=False))

                csv_data = filtered.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download filtered finance log as CSV",
                    data=csv_data,
                    file_name="sff_finance_filtered.csv",
                    mime="text/csv",
                )

        # ADD ENTRY
        with tab_add:
            st.subheader("Add New Finance Entry")

            with st.form("add_finance_form"):
                col1, col2 = st.columns(2)
                with col1:
                    entry_date = st.date_input("Date", value=datetime.today())
                    entry_type = st.selectbox("Type", ["Income", "Expense"])
                    category = st.text_input("Category (e.g., Donation, Grant, Operating)")
                with col2:
                    project_options = ["General"]
                    if not projects_df.empty:
                        project_options += sorted(
                            projects_df["project"].dropna().unique().tolist()
                        )
                    project = st.selectbox("Project", project_options)
                    amount_input = st.number_input(
                        "Amount (positive number)", min_value=0.0, step=100.0
                    )
                description = st.text_area("Description")

                submitted = st.form_submit_button("Save Finance Entry")
                if submitted:
                    if amount_input == 0:
                        st.error("Amount must be greater than zero.")
                    else:
                        sign = 1 if entry_type == "Income" else -1
                        new_row = {
                            "date": pd.to_datetime(entry_date),
                            "category": category if category else "Uncategorized",
                            "type": entry_type,
                            "project": project,
                            "description": description,
                            "amount": sign * float(amount_input),
                        }
                        updated = pd.concat(
                            [finance_df, pd.DataFrame([new_row])],
                            ignore_index=True,
                        )
                        save_finance(updated)
                        st.success(
                            "Finance entry saved! Go to Dashboard to see updated metrics."
                        )

        # IMPORT FROM CSV
        with tab_import:
            st.subheader("Import from Bank/Broker CSV")

            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded is not None:
                raw = pd.read_csv(uploaded)
                st.write("Preview of uploaded file:")
                st.dataframe(raw.head())

                cols = raw.columns.tolist()
                c1, c2, c3 = st.columns(3)
                with c1:
                    date_col = st.selectbox("Date column", cols)
                with c2:
                    desc_col = st.selectbox("Description column", cols)
                with c3:
                    amount_col = st.selectbox("Amount column", cols)

                default_category = st.text_input("Default Category", "Imported")
                default_project = st.text_input("Default Project", "General")

                if st.button("Import rows"):
                    df_imp = pd.DataFrame()
                    df_imp["date"] = pd.to_datetime(raw[date_col], errors="coerce")
                    df_imp["description"] = raw[desc_col].astype(str)
                    df_imp["amount"] = pd.to_numeric(raw[amount_col], errors="coerce")

                    df_imp["type"] = df_imp["amount"].apply(
                        lambda x: "Income" if x > 0 else "Expense"
                    )
                    df_imp["category"] = default_category
                    df_imp["project"] = default_project

                    df_imp = df_imp[
                        ["date", "category", "type", "project", "description", "amount"]
                    ].dropna(subset=["date", "amount"])

                    updated = pd.concat([finance_df, df_imp], ignore_index=True)
                    save_finance(updated)
                    st.success(f"Imported {len(df_imp)} rows into finance log.")

# ---- PROJECTS ----
elif page == "Projects":
    st.title("📁 SFF Projects & Grants")

    tab_view, tab_add, tab_update, tab_score = st.tabs(
        ["View Projects", "Add Project", "Update Spend", "Grant Scoring"]
    )

    # VIEW
    with tab_view:
        if projects_df.empty:
            st.info("No projects yet.")
        else:
            st.dataframe(projects_df)

    # ADD
    with tab_add:
        if role != "admin":
            st.info("Only admin can add new projects.")
        else:
            st.subheader("Add New Project")
            with st.form("add_project_form"):
                project_name = st.text_input("Project Name")
                category = st.text_input("Category (Education, R&D, Health, etc.)")
                status = st.selectbox("Status", ["Planned", "Active", "Completed"])
                budget = st.number_input("Budget", min_value=0.0, step=1000.0)
                notes = st.text_area("Notes")

                submitted = st.form_submit_button("Save Project")
                if submitted:
                    if not project_name:
                        st.error("Project name is required.")
                    else:
                        new_row = {
                            "project": project_name,
                            "category": category if category else "Uncategorized",
                            "status": status,
                            "budget": float(budget),
                            "spent": 0.0,
                            "notes": notes,
                        }
                        updated = pd.concat(
                            [projects_df, pd.DataFrame([new_row])],
                            ignore_index=True,
                        )
                        save_projects(updated)
                        st.success("Project added!")

    # UPDATE SPEND
    with tab_update:
        if projects_df.empty:
            st.info("No projects to update.")
        else:
            st.subheader("Update Project Spend")
            project_names = projects_df["project"].tolist()
            selected_project = st.selectbox("Select Project", project_names)

            current_row = projects_df[projects_df["project"] == selected_project].iloc[0]
            st.write(
                f"Current spent: **${current_row['spent']:,.0f}** / "
                f"Budget: **${current_row['budget']:,.0f}**"
            )

            additional_spend = st.number_input(
                "Add to Spent Amount", min_value=0.0, step=500.0
            )

            status_options = ["Planned", "Active", "Completed"]
            status_index = (
                status_options.index(current_row["status"])
                if current_row["status"] in status_options
                else 0
            )
            new_status = st.selectbox(
                "Update Status", status_options, index=status_index
            )

            if st.button("Update Project"):
                idx = projects_df[projects_df["project"] == selected_project].index[0]
                projects_df.loc[idx, "spent"] = float(current_row["spent"]) + float(
                    additional_spend
                )
                projects_df.loc[idx, "status"] = new_status
                save_projects(projects_df)
                st.success("Project updated!")

    # GRANT SCORING
    with tab_score:
        st.subheader("Grant Scoring & Prioritization")
        if projects_df.empty:
            st.info("No projects to score yet.")
        else:
            st.markdown(
                "This scoring favors **Active, underfunded** projects "
                "so you know where to deploy money next."
            )
            underfund_weight = st.slider(
                "Weight on 'underfunded' (0–100)", min_value=0, max_value=100, value=60
            )

            def score_project(row):
                status_weights = {"Planned": 30, "Active": 80, "Completed": 20}
                sw = status_weights.get(row.get("status", ""), 50)
                budget = row.get("budget", 0) or 0
                spent = row.get("spent", 0) or 0
                if budget > 0:
                    util = min(max(spent / budget, 0), 1)
                    underfund = 1 - util
                else:
                    underfund = 0.5
                uf_score = underfund * underfund_weight
                return sw + uf_score

            scored = projects_df.copy()
            scored["score"] = scored.apply(score_project, axis=1)
            scored = scored.sort_values("score", ascending=False)

            st.dataframe(
                scored[
                    ["project", "category", "status", "budget", "spent", "score"]
                ]
            )

# ---- SCENARIOS ----
elif page == "Scenarios":
    st.title("📈 Scenario Planning – What If?")

    if finance_df.empty:
        st.info("Add some finance data first to run scenarios.")
    else:
        base_kpis = compute_kpis(finance_df)

        col1, col2 = st.columns(2)
        with col1:
            income_change = st.slider(
                "Change in Income (%)", min_value=-50, max_value=50, value=0, step=5
            )
        with col2:
            expense_change = st.slider(
                "Change in Expenses (%)", min_value=-50, max_value=50, value=0, step=5
            )

        scen = finance_df.copy()
        inc_mask = scen["type"] == "Income"
        exp_mask = scen["type"] == "Expense"

        scen.loc[inc_mask, "amount"] *= (1 + income_change / 100)
        scen.loc[exp_mask, "amount"] *= (1 + expense_change / 100)

        scen_kpis = compute_kpis(scen)

        st.markdown("### Comparison")
        colb1, colb2 = st.columns(2)
        with colb1:
            st.markdown("**Base Case**")
            st.write(f"Total Income: ${base_kpis['total_income']:,.0f}")
            st.write(f"Total Expenses: ${base_kpis['total_expense'] * -1:,.0f}")
            st.write(f"Net Cash: ${base_kpis['net_cash']:,.0f}")
            st.write(f"Program Ratio: {base_kpis['program_ratio'] * 100:.1f}%")
        with colb2:
            st.markdown("**Scenario Case**")
            st.write(f"Total Income: ${scen_kpis['total_income']:,.0f}")
            st.write(f"Total Expenses: ${scen_kpis['total_expense'] * -1:,.0f}")
            st.write(f"Net Cash: ${scen_kpis['net_cash']:,.0f}")
            st.write(f"Program Ratio: {scen_kpis['program_ratio'] * 100:.1f}%")

        st.markdown("### Cash Balance in Scenario")
        scen_sorted = scen.sort_values("date").copy()
        scen_sorted["cumulative_cash"] = scen_sorted["amount"].cumsum()
        st.line_chart(scen_sorted.set_index("date")["cumulative_cash"])

# ---- CONTACTS ----
elif page == "Contacts":
    st.title("👥 Contacts (Donors, Partners, Advisors)")

    tab_view, tab_add = st.tabs(["View Contacts", "Add Contact"])

    with tab_view:
        if contacts_df.empty:
            st.info("No contacts yet.")
        else:
            st.dataframe(contacts_df)

    with tab_add:
        if role != "admin":
            st.info("Only admin can add or edit contacts.")
        else:
            st.subheader("Add New Contact")
            with st.form("add_contact_form"):
                name = st.text_input("Name")
                ctype = st.selectbox(
                    "Type", ["Donor", "Partner", "Advisor", "Other"]
                )
                org = st.text_input("Organization")
                email = st.text_input("Email")
                phone = st.text_input("Phone")
                notes = st.text_area("Notes")

                submitted = st.form_submit_button("Save Contact")
                if submitted:
                    if not name:
                        st.error("Name is required.")
                    else:
                        new_row = {
                            "name": name,
                            "type": ctype,
                            "organization": org,
                            "email": email,
                            "phone": phone,
                            "notes": notes,
                        }
                        updated = pd.concat(
                            [contacts_df, pd.DataFrame([new_row])],
                            ignore_index=True,
                        )
                        save_contacts(updated)
                        st.success("Contact saved!")

# ---- NOTES ----
elif page == "Notes":
    st.title("📝 Meeting Notes & Board Decisions")

    tab_view, tab_add = st.tabs(["View Notes", "Add Note"])

    with tab_view:
        if notes_df.empty:
            st.info("No notes yet.")
        else:
            st.dataframe(notes_df.sort_values("date", ascending=False))

    with tab_add:
        if role not in ("admin", "board"):
            st.info("Only admin/board can record notes.")
        else:
            st.subheader("Add New Note")
            with st.form("add_note_form"):
                date_val = st.date_input("Date", value=datetime.today())
                mtype = st.selectbox(
                    "Meeting Type", ["Board", "Family", "Operations", "Other"]
                )
                title = st.text_input("Title")
                summary = st.text_area("Summary / Discussion")
                decisions = st.text_area("Decisions / Action Items")
                attendees = st.text_area("Attendees")

                submitted = st.form_submit_button("Save Note")
                if submitted:
                    if not title:
                        st.error("Title is required.")
                    else:
                        new_row = {
                            "date": pd.to_datetime(date_val),
                            "meeting_type": mtype,
                            "title": title,
                            "summary": summary,
                            "decisions": decisions,
                            "attendees": attendees,
                        }
                        updated = pd.concat(
                            [notes_df, pd.DataFrame([new_row])],
                            ignore_index=True,
                        )
                        save_notes(updated)
                        st.success("Note saved!")

# ---- GOALS ----
elif page == "Goals":
    st.title("🎯 Goals & OKRs")

    tab_view, tab_add = st.tabs(["View Goals", "Add Goal"])

    with tab_view:
        if goals_df.empty:
            st.info("No goals yet.")
        else:
            display = goals_df.copy()
            display["progress"] = display["progress"].fillna(0)
            st.dataframe(display)

    with tab_add:
        if role != "admin":
            st.info("Only admin can add or edit goals.")
        else:
            st.subheader("Add New Goal / OKR")
            with st.form("add_goal_form"):
                goal_text = st.text_area("Goal")
                owner = st.text_input("Owner", value="Arav")
                area = st.selectbox(
                    "Area", ["Education", "Health", "R&D", "Operations", "Other"]
                )
                target_date = st.date_input("Target Date")
                status = st.selectbox(
                    "Status",
                    ["Not started", "In progress", "At risk", "Completed"],
                )
                progress = st.slider("Progress (%)", 0, 100, 0)
                notes = st.text_area("Notes")

                submitted = st.form_submit_button("Save Goal")
                if submitted:
                    if not goal_text:
                        st.error("Goal description is required.")
                    else:
                        new_row = {
                            "goal": goal_text,
                            "owner": owner,
                            "area": area,
                            "target_date": pd.to_datetime(target_date),
                            "status": status,
                            "progress": progress,
                            "notes": notes,
                        }
                        updated = pd.concat(
                            [goals_df, pd.DataFrame([new_row])],
                            ignore_index=True,
                        )
                        save_goals(updated)
                        st.success("Goal saved!")

# ---- BOARD VIEW ----
elif page == "Board View":
    st.title("📑 Board View – SFF Snapshot")

    if finance_df.empty and projects_df.empty:
        st.info("Add some finance entries and projects to populate the board view.")
    else:
        st.markdown("### 1. High-Level Financials")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Income (YTD)", f"${kpis['total_income']:,.0f}")
        with col2:
            st.metric("Total Spend (YTD)", f"${kpis['total_expense'] * -1:,.0f}")
        with col3:
            st.metric("Net Cash Position", f"${kpis['net_cash']:,.0f}")

        st.markdown("### 2. Program vs Operating")
        col4, col5 = st.columns(2)
        with col4:
            st.write(f"Program Spend: **${kpis['program_spend']:,.0f}**")
            st.write(f"Operating Spend: **${kpis['operating_spend']:,.0f}**")
            st.write(f"Program Ratio: **{kpis['program_ratio'] * 100:.1f}%**")
        with col5:
            if kpis["program_spend"] + kpis["operating_spend"] > 0:
                ratio_df = pd.DataFrame(
                    {
                        "Category": ["Program", "Operating"],
                        "Amount": [kpis["program_spend"], kpis["operating_spend"]],
                    }
                ).set_index("Category")
                st.bar_chart(ratio_df["Amount"])
            else:
                st.info("No expense data yet to chart.")

        st.markdown("### 3. Projects Overview")
        if not projects_df.empty:
            summary = projects_df.groupby("status").agg(
                num_projects=("project", "count"),
                total_budget=("budget", "sum"),
                total_spent=("spent", "sum"),
            )
            st.dataframe(summary)

            st.markdown("**Top 5 projects by budget:**")
            top5 = projects_df.sort_values("budget", ascending=False).head(5)
            st.dataframe(
                top5[["project", "category", "status", "budget", "spent"]]
            )
        else:
            st.info("No projects in the system yet.")

        st.markdown("---")
        st.markdown("### PDF Export")

        if st.button("Generate Board Report PDF"):
            pdf_bytes = generate_board_report_pdf(finance_df, projects_df)
            st.download_button(
                "Download Board Report PDF",
                data=pdf_bytes,
                file_name="sff_board_report.pdf",
                mime="application/pdf",
            )

# ---- SETTINGS ----
elif page == "Settings":
    st.title("⚙️ SFF HQ – Settings & Info")

    st.markdown(
        """
        **Features currently implemented:**

        - 🔐 Login for admin + board
        - 💸 Finance Log (view, add, CSV import)
        - 📁 Projects with spend tracking & grant scoring
        - 👥 Contacts (donors, partners, advisors)
        - 📝 Notes (meeting notes & decisions)
        - 🎯 Goals / OKRs
        - 📈 Scenario planning (income/expense shocks)
        - 📑 Board View with PDF export
        """
    )

    st.caption("Built for the Soni Family Foundation – internal use only.")
