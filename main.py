import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import sqlite3
import math
from PIL import Image
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Database setup
DB_FILE = 'employees.db'


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Create the employees table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS employees
                 (id INTEGER PRIMARY KEY,
                  full_name TEXT,
                  email TEXT UNIQUE,
                  mobile TEXT,
                  date_of_birth TEXT,
                  photo BLOB,
                  performance_score FLOAT)''')

    conn.commit()
    conn.close()


def add_employee_to_db(employee, photo):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''INSERT INTO employees (full_name, email, mobile, date_of_birth, photo, performance_score)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (employee['Full Name'], employee['Email'], employee['Mobile'], employee['Date Of Birth'],
               photo, employee['Performance Score']))
    conn.commit()
    conn.close()


def get_employees_from_db(page, per_page, search_name='', search_email='', search_dob=None):
    conn = sqlite3.connect(DB_FILE)
    offset = (page - 1) * per_page
    query = "SELECT id, full_name, email, mobile, date_of_birth, photo, performance_score FROM employees WHERE 1=1"
    params = []
    if search_name:
        query += " AND full_name LIKE ?"
        params.append(f"%{search_name}%")
    if search_email:
        query += " AND email LIKE ?"
        params.append(f"%{search_email}%")
    if search_dob:
        query += " AND date_of_birth = ?"
        params.append(search_dob.strftime("%d/%m/%Y"))
    query += f" ORDER BY id LIMIT {per_page} OFFSET {offset}"
    df = pd.read_sql_query(query, conn, params=params)

    count_query = "SELECT COUNT(*) FROM employees WHERE 1=1"
    count_params = []
    if search_name:
        count_query += " AND full_name LIKE ?"
        count_params.append(f"%{search_name}%")
    if search_email:
        count_query += " AND email LIKE ?"
        count_params.append(f"%{search_email}%")
    if search_dob:
        count_query += " AND date_of_birth = ?"
        count_params.append(search_dob.strftime("%d/%m/%Y"))
    total_employees = pd.read_sql_query(count_query, conn, params=count_params).iloc[0, 0]

    conn.close()
    return df, total_employees


def get_employee_by_id(employee_id):
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT id, full_name, email, mobile, date_of_birth, photo, performance_score FROM employees WHERE id = ?"
    df = pd.read_sql_query(query, conn, params=(employee_id,))
    conn.close()
    if df.empty:
        return None
    return df.iloc[0]


def update_employee_in_db(employee_id, employee, photo=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if photo is not None:
        c.execute('''UPDATE employees 
                     SET full_name = ?, email = ?, mobile = ?, date_of_birth = ?, photo = ?, performance_score = ?
                     WHERE id = ?''',
                  (employee['Full Name'], employee['Email'], employee['Mobile'], employee['Date Of Birth'],
                   photo, employee['Performance Score'], employee_id))
    else:
        c.execute('''UPDATE employees 
                     SET full_name = ?, email = ?, mobile = ?, date_of_birth = ?, performance_score = ?
                     WHERE id = ?''',
                  (employee['Full Name'], employee['Email'], employee['Mobile'], employee['Date Of Birth'],
                   employee['Performance Score'], employee_id))
    conn.commit()
    conn.close()


def delete_employee_from_db(employee_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('DELETE FROM employees WHERE id = ?', (employee_id,))
    conn.commit()
    conn.close()


def resize_image(image, size=(100, 100)):
    img = Image.open(image)
    img.thumbnail(size)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def image_to_base64(image_binary):
    return base64.b64encode(image_binary).decode('utf-8')


def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def predict_performance():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM employees", conn)
    conn.close()

    if len(df) < 10:  # We need at least some data to make predictions
        st.warning("Not enough data to make predictions. Please add more employees.")
        return

    df['age'] = df['date_of_birth'].apply(lambda x: calculate_age(datetime.strptime(x, "%d/%m/%Y").date()))

    X = df[['age']]
    y = df['performance_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    st.subheader("Performance Prediction Model")
    st.write(f"Model R² Score: {model.score(X_test_scaled, y_test):.2f}")

    st.subheader("Feature Importance")
    st.write("Age importance:", model.coef_[0])

    st.subheader("Predictions")
    age_range = range(20, 61)
    predictions = model.predict(scaler.transform([[age] for age in age_range]))

    prediction_df = pd.DataFrame({'Age': age_range, 'Predicted Performance': predictions})
    st.line_chart(prediction_df.set_index('Age'))

    st.subheader("Interpret the Results")
    st.write(
        "This simple model predicts employee performance based on age. The line chart shows the predicted performance score for different ages. Remember, this is a very simplified model and real-world performance depends on many more factors!")


def main():
    st.set_page_config(page_title="Employee Management System", layout="wide")

    st.title("Employee Management System")

    # Initialize database
    init_db()

    # Initialize session state variables
    if 'page' not in st.session_state:
        st.session_state.page = 1
    if 'editing' not in st.session_state:
        st.session_state.editing = None
    if 'deleting' not in st.session_state:
        st.session_state.deleting = None
    if 'deleting_name' not in st.session_state:
        st.session_state.deleting_name = None

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["View Employees", "Add Employee", "Performance Predictions"])

    if page == "View Employees":
        view_employees()
    elif page == "Add Employee":
        add_employee()
    elif page == "Performance Predictions":
        predict_performance()

def view_employees():
    st.header("Employee List")

    # Initialize session state for search parameters if not exists
    if 'search_name' not in st.session_state:
        st.session_state.search_name = ''
    if 'search_email' not in st.session_state:
        st.session_state.search_email = ''
    if 'search_dob' not in st.session_state:
        st.session_state.search_dob = None

    # Search form
    st.subheader("Search Employees")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        search_name = st.text_input("Name", value=st.session_state.search_name)
    with col2:
        search_dob = st.date_input("Date of Birth", value=st.session_state.search_dob, min_value=date(1900, 1, 1), max_value=date.today())
    with col3:
        search_email = st.text_input("Email", value=st.session_state.search_email)
    with col4:
        search_mobile = st.text_input("Mobile")
    with col5:
        st.write("")
        st.write("")
        search_button = st.button("🔍")
        clear_button = st.button("Clear Search")

    # Update session state when search button is clicked
    if search_button:
        st.session_state.search_name = search_name
        st.session_state.search_email = search_email
        st.session_state.search_dob = search_dob
        st.session_state.page = 1  # Reset to first page on new search
    elif clear_button:
        st.session_state.search_name = ''
        st.session_state.search_email = ''
        st.session_state.search_dob = None
        st.session_state.page = 1
        st.rerun()

    # Pagination
    per_page = 6
    if 'page' not in st.session_state:
        st.session_state.page = 1
    page = st.session_state.page

    # Fetch employees from database with pagination and search
    df, total_employees = get_employees_from_db(
        page,
        per_page,
        st.session_state.search_name,
        st.session_state.search_email,
        st.session_state.search_dob
    )

    if total_employees == 0:
        st.info("No employees found matching the search criteria.")
    else:
        # Display employee table
        for _, row in df.iterrows():
            col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 2, 2, 2, 2])
            with col1:
                if row['photo']:
                    st.image(row['photo'], width=50)
                else:
                    st.image("https://via.placeholder.com/50", width=50)
            with col2:
                st.write(row['full_name'])
            with col3:
                st.write(row['email'])
            with col4:
                st.write(row['mobile'])
            with col5:
                st.write(row['date_of_birth'])
            with col6:
                col6_1, col6_2 = st.columns(2)
                with col6_1:
                    if st.button('✏️', key=f'edit_{row["id"]}'):
                        st.session_state.editing = row['id']
                        st.session_state.deleting = None
                        st.rerun()
                with col6_2:
                    if st.button('🗑️', key=f'delete_{row["id"]}'):
                        st.session_state.deleting = row['id']
                        st.session_state.deleting_name = row['full_name']
                        st.session_state.editing = None
                        st.rerun()

        total_pages = math.ceil(total_employees / per_page)
        st.write(f"Showing {(page-1)*per_page + 1} to {min(page*per_page, total_employees)} out of {total_employees}")

        # Pagination controls
        cols = st.columns(9)
        if cols[0].button("<<", key="first"):
            st.session_state.page = 1
            st.rerun()
        if cols[1].button("<", key="prev"):
            st.session_state.page = max(1, page - 1)
            st.rerun()
        for i in range(1, 6):
            if cols[i+1].button(str(i), key=f"page_{i}"):
                st.session_state.page = i
                st.rerun()
        if cols[7].button(">", key="next"):
            st.session_state.page = min(total_pages, page + 1)
            st.rerun()
        if cols[8].button(">>", key="last"):
            st.session_state.page = total_pages
            st.rerun()

    # Handle editing
    if 'editing' in st.session_state and st.session_state.editing is not None:
        edit_employee(st.session_state.editing)

    # Handle deleting
    if 'deleting' in st.session_state and st.session_state.deleting is not None:
        st.warning(f"Are you sure you want to delete {st.session_state.deleting_name}?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, delete"):
                delete_employee_from_db(st.session_state.deleting)
                st.success(f"Employee {st.session_state.deleting_name} has been deleted.")
                st.session_state.deleting = None
                st.session_state.deleting_name = None
                st.rerun()
        with col2:
            if st.button("No, cancel"):
                st.session_state.deleting = None
                st.session_state.deleting_name = None
                st.rerun()

def edit_employee(employee_id):
    st.subheader("Edit Employee")
    employee = get_employee_by_id(employee_id)

    if employee is None:
        st.error("Employee not found.")
        return

    with st.form("edit_employee_form"):
        full_name = st.text_input("Full Name", value=employee['full_name'])
        email = st.text_input("Email", value=employee['email'])
        mobile = st.text_input("Mobile", value=employee['mobile'])
        date_of_birth = st.date_input("Date of Birth",
                                      value=datetime.strptime(employee['date_of_birth'], "%d/%m/%Y").date())
        performance_score = st.slider("Performance Score", 0.0, 10.0, float(employee['performance_score']), 0.1)
        new_photo = st.file_uploader("Upload New Photo (optional)", type=["jpg", "png", "jpeg"])

        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Update Employee"):
                updated_employee = {
                    "Full Name": full_name,
                    "Email": email,
                    "Mobile": mobile,
                    "Date Of Birth": date_of_birth.strftime("%d/%m/%Y"),
                    "Performance Score": performance_score
                }
                if new_photo:
                    photo_binary = resize_image(new_photo)
                    update_employee_in_db(employee_id, updated_employee, photo_binary)
                else:
                    update_employee_in_db(employee_id, updated_employee)
                st.success(f"Employee {full_name} updated successfully!")
                st.session_state.editing = None
                st.rerun()
        with col2:
            if st.form_submit_button("Cancel"):
                st.session_state.editing = None
                st.rerun()


def add_employee():
    st.header("Add New Employee")

    with st.form("add_employee_form"):
        col1, col2 = st.columns(2)

        with col1:
            first_name = st.text_input("First Name")
            email = st.text_input("Email")
            min_date = date(1900, 1, 1)
            max_date = date.today()
            date_of_birth = st.date_input("Date of Birth", min_value=min_date, max_value=max_date)

        with col2:
            last_name = st.text_input("Last Name")
            mobile = st.text_input("Mobile")
            performance_score = st.slider("Performance Score", 0.0, 10.0, 5.0, 0.1)
            photo = st.file_uploader("Upload Photo", type=["jpg", "png", "jpeg"])

        submitted = st.form_submit_button("Add Employee")
        if submitted:
            full_name = f"{first_name} {last_name}"
            dob_str = date_of_birth.strftime("%d/%m/%Y")

            new_employee = {
                "Full Name": full_name,
                "Email": email,
                "Mobile": mobile,
                "Date Of Birth": dob_str,
                "Performance Score": performance_score
            }

            try:
                if photo:
                    photo_binary = resize_image(photo)
                else:
                    photo_binary = None
                add_employee_to_db(new_employee, photo_binary)
                st.success(f"Employee {full_name} added successfully!")

                # Display the added employee's details
                st.subheader("Added Employee Details:")
                st.write(f"Full Name: {full_name}")
                st.write(f"Email: {email}")
                st.write(f"Mobile: {mobile}")
                st.write(f"Date of Birth: {dob_str}")
                st.write(f"Performance Score: {performance_score}")

                if photo:
                    st.image(photo, caption="Uploaded Photo", width=100)
            except sqlite3.IntegrityError:
                st.error("An employee with this email already exists. Please use a different email.")


if __name__ == "__main__":
    main()