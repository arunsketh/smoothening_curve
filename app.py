import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime

# --- 1. Define Core Functions ---

def get_parametric_interpolated_data(data_str, path_step_size):
    """
    Reads data from a string, performs parametric interpolation along the path length,
    and returns:
      - original_data: (original_x, original_y) in their exact input order
      - new_data: (new_x_values, new_y_values) evenly spaced along the path
      - error: None or message
    """
    # Parse input into list of (x, y) in original sequential order
    data_lines = data_str.strip().splitlines()
    data = []
    for i, line in enumerate(data_lines):
        line = line.strip()
        if not line:
            continue
        try:
            cleaned_line = "".join(
                c for c in line if c.isdigit() or c == "." or c == "," or c == "-" or c.isspace()
            )
            parts = cleaned_line.replace(",", " ").split()
            if len(parts) != 2:
                st.error(f"⚠️ Input Error: Line {i+1} must contain exactly two numbers.")
                return None, None, None
            x, y = map(float, parts)
            data.append((x, y))
        except ValueError:
            st.error(
                f"⚠️ Input Error: Could not convert values on line {i+1}. Please check your format."
            )
            return None, None, None

    if len(data) < 2:
        st.error("⚠️ Input Error: At least two data points are needed for interpolation.")
        return None, None, None

    # Keep original order intact to preserve shapes like loops or U-curves
    original_x = np.array([item[0] for item in data])
    original_y = np.array([item[1] for item in data])

    # 1. Calculate the distance between consecutive points (dx, dy)
    dx = np.diff(original_x)
    dy = np.diff(original_y)
    
    # 2. Calculate the segment lengths and cumulative distance (S) along the curve
    distances = np.sqrt(dx**2 + dy**2)
    cumulative_s = np.concatenate(([0], np.cumsum(distances)))
    total_length = cumulative_s[-1]

    # 3. Create a new evenly spaced grid along the total path length
    if path_step_size >= total_length:
        st.error("⚠️ Step size is larger than the entire curve. Choose a smaller step size.")
        return None, None, None
        
    new_s = np.arange(0, total_length, path_step_size)

    # Ensure the exact final endpoint of the curve is included
    if not np.isclose(new_s[-1], total_length, atol=1e-6):
        new_s = np.append(new_s, total_length)

    # 4. Interpolate X and Y individually against the cumulative path distance
    new_x_values = np.interp(new_s, cumulative_s, original_x)
    new_y_values = np.interp(new_s, cumulative_s, original_y)

    return (original_x, original_y), (new_x_values, new_y_values), None


# --- 2. Build the Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title("📈 Parametric Curve Optimisation GUI")
st.markdown("Ideal for complex shapes, U-curves, hysteresis loops, and aerofoils.")

st.sidebar.header("Controls")

# Default data replaced with a small U-shape example similar to user data
default_data = (
    "-2.10, -10.27\n-4.07, -10.44\n-6.09, -12.17\n-8.04, -15.76\n-7.52, -18.36\n"
    "-5.10, -21.42\n-2.03, -21.98\n0.00, -22.58\n2.03, -21.98\n5.10, -21.42\n"
    "7.52, -18.36\n8.04, -15.76\n6.09, -12.17\n4.07, -10.44\n2.10, -10.27"
)

data_str = st.sidebar.text_area(
    "Input Data (X, Y per line)",
    value=default_data,
    height=250,
    help="Paste data here. Values can be separated by commas or spaces. Supports negatives."
)

st.sidebar.markdown("---")
path_step_size = st.sidebar.number_input(
    "Path Step Size (Distance between points)",
    min_value=0.01,
    value=0.50,
    step=0.10,
    format="%.2f",
    help="Determines the physical distance between new points along the curve's path."
)

# Single placeholder for all output
output_placeholder = st.empty()

# --- 3. Initialize Session State ---
if "results_data" not in st.session_state:
    st.session_state.results_data = None
    st.session_state.original_data = None
    st.session_state.generation_time = None

if "just_generated" not in st.session_state:
    st.session_state.just_generated = False

# --- 4. Handle Button Click to Run Calculation and Save State ---
if st.sidebar.button("Generate Curve", type="primary"):
    output_placeholder.empty()
    if not data_str.strip():
        st.warning("Please provide input data.")
        st.session_state.results_data = None
        st.session_state.just_generated = False
    else:
        original_data, new_data, error = get_parametric_interpolated_data(data_str, path_step_size)
        if error:
            st.error(error)
            st.session_state.results_data = None
            st.session_state.just_generated = False
        else:
            st.session_state.original_data = original_data
            st.session_state.results_data = new_data
            st.session_state.generation_time = time.time()
            st.session_state.just_generated = True

# --- 5. Display the Output inside the placeholder if it Exists ---
if st.session_state.results_data is not None:
    with output_placeholder.container():
        generation_dt = datetime.fromtimestamp(st.session_state.generation_time)
        st.success(f"✅ Success! Parametric curve generated at {generation_dt.strftime('%H:%M:%S')}.")

        col1, col2 = st.columns([1, 2])

        original_data = st.session_state.original_data
        new_data = st.session_state.results_data

        with col1:
            st.subheader("Interpolated Results")
            # Prepare CSV
            result_lines = ["X,Y"]
            for x, y in zip(new_data[0], new_data[1]):
                result_lines.append(f"{x:.4f},{y:.4f}")
            result_csv = "\n".join(result_lines)

            if st.session_state.just_generated or "output_text_area" not in st.session_state:
                st.session_state["output_text_area"] = result_csv

            st.text_area(
                "Output Data",
                st.session_state["output_text_area"],
                height=300,
                key="output_text_area",
            )

            st.download_button(
                label="Download data as CSV",
                data=st.session_state["output_text_area"],
                file_name="parametric_interpolated_data.csv",
                mime="text/csv",
            )

        with col2:
            st.subheader("Plot")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot original as dashed line with distinct markers
            ax.plot(original_data[0], original_data[1], 'r--', marker='s', markersize=6, alpha=0.5, label='Original Path')
            
            # Plot new interpolated data over top
            ax.plot(new_data[0], new_data[1], 'b-', marker='o', markersize=4, label='Parametric Interpolated')
            
            ax.set_xlabel('X values')
            ax.set_ylabel('Y values')
            ax.set_title('Parametric Curve Interpolation')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Keep aspect ratio equal if it's a physical shape to avoid distortion
            # ax.set_aspect('equal', adjustable='box') 
            
            st.pyplot(fig)

        # Reset flag
        st.session_state.just_generated = False
