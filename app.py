import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io

# --- 1. Define Core Functions (This section is unchanged) ---

def get_interpolated_data(data_str, step_size):
    """
    Reads data from a string, performs interpolation, and returns the results.
    """
    # Convert input string to a list of (x, y) tuples
    data_lines = data_str.strip().splitlines()
    data = []
    for i, line in enumerate(data_lines):
        line = line.strip()
        if not line: continue
        try:
            # Flexible parsing for comma or space separated values
            cleaned_line = "".join(c for c in line if c.isdigit() or c == '.' or c == ',' or c.isspace())
            parts = cleaned_line.replace(',', ' ').split()
            if len(parts) != 2:
                st.error(f"⚠️ Input Error: Line {i+1} must contain exactly two numbers.")
                return None, None, None
            x, y = map(float, parts)
            data.append((x, y))
        except ValueError:
            st.error(f"⚠️ Input Error: Could not convert values on line {i+1}. Please check your format.")
            return None, None, None

    if len(data) < 2:
        st.error("⚠️ Input Error: At least two data points are needed for interpolation.")
        return None, None, None

    # IMPORTANT: Sort data based on y-values for correct interpolation
    data.sort(key=lambda item: item[1])
    original_x = np.array([item[0] for item in data])
    original_y = np.array([item[1] for item in data])

    # Define the range for new y values
    min_y, max_y = original_y[0], original_y[-1]
    new_y_values = np.arange(min_y, max_y + step_size * 1e-9, step_size)

    # Compute corresponding x values using numpy's interpolation
    new_x_values = np.interp(new_y_values, original_y, original_x)

    return (original_x, original_y), (new_x_values, new_y_values), None


# --- 2. Build the Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title("📈 Curve Optimisation GUI")

# Use a sidebar for inputs to keep the main area clean
st.sidebar.header("Controls")

default_data = (
    "        0.,     0.\n   2.20284, 0.0014\n   12.1156, 0.0056\n   16.5213,  0.007\n   "
    "20.9269, 0.0083\n   23.1298,  0.009\n   25.3326, 0.0098\n   27.5355, 0.0106\n   "
    "861.309, 0.1038\n   865.715,  0.104"
)

data_str = st.sidebar.text_area(
    "Input Data (X, Y per line)", value=default_data, height=250,
    help="Paste your data here. Values can be separated by commas or spaces."
)

step_size = st.sidebar.number_input(
    "Step Size for Y-axis (Stress)", min_value=0.0001, value=0.005, step=0.001, format="%.4f"
)

# --- 3. Initialize Session State ---
# This runs once per session to make sure our variables exist.
if 'results_data' not in st.session_state:
    st.session_state.results_data = None
    st.session_state.original_data = None

# --- 4. Handle Button Click to Run Calculation and Save State ---
if st.sidebar.button("Generate Curve", type="primary"):
    if not data_str.strip():
        st.warning("Please provide input data.")
        st.session_state.results_data = None # Clear previous results
    else:
        original_data, new_data, error = get_interpolated_data(data_str, step_size)
        if error:
            st.error(error)
            st.session_state.results_data = None # Clear previous results on error
        else:
            # On success, save the data to the session state
            st.session_state.original_data = original_data
            st.session_state.results_data = new_data

# --- 5. Display the Output if it Exists in Session State ---
# This block will run on EVERY page rerun, ensuring the output stays visible.
if st.session_state.results_data is not None:
    st.success("✅ Success! Interpolation complete.")
    col1, col2 = st.columns([1, 2])

    # Retrieve data from session state for display
    original_data = st.session_state.original_data
    new_data = st.session_state.results_data

    with col1:
        st.subheader("Interpolated Results")
        result_lines = ["X (Strain),Y (Stress)"]
        for x, y in zip(new_data[0], new_data[1]):
            result_lines.append(f"{x:.2f},{y:.3f}")
        result_csv = "\n".join(result_lines)
        
        st.text_area("Output Data", result_csv, height=300, key="output_text_area")

        st.download_button(
           label="Download data as CSV",
           data=result_csv,
           file_name='interpolated_data.csv',
           mime='text/csv',
        )

    with col2:
        st.subheader("Plot")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(original_data[1], original_data[0], 'r-', marker='.', markersize=8, label='Input Data')
        ax.plot(new_data[1], new_data[0], 'bo', markersize=4, label='Interpolated Data')
        ax.set_xlabel('Y values (Stress)')
        ax.set_ylabel('X values (Strain)')
        ax.set_title('Input and Interpolated Data')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
