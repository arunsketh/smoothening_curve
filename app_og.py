import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime

# --- 1. Define Core Functions ---

def get_interpolated_data(data_str, step_size):
    """
    Reads data from a string, performs interpolation, and returns:
      - original_data: (original_x_sorted_by_y, original_y_sorted_by_y)
      - new_data: (new_x_values_with_endpoints, new_y_values_with_endpoints)
      - error: None or message
    Also ensures the first and last input points are retained in the output.
    """
    # Parse input into list of (x, y) in original order
    data_lines = data_str.strip().splitlines()
    data = []
    for i, line in enumerate(data_lines):
        line = line.strip()
        if not line:
            continue
        try:
            cleaned_line = "".join(
                c for c in line if c.isdigit() or c == "." or c == "," or c.isspace()
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

    # Retain the first and last points from the original input order
    first_raw = data[0]
    last_raw = data[-1]

    # Sort by Y for monotonic interpolation
    data_sorted = sorted(data, key=lambda item: item[1])
    original_x = np.array([item[0] for item in data_sorted])
    original_y = np.array([item[1] for item in data_sorted])

    # Define Y grid over [min_y, max_y] with step_size
    min_y, max_y = original_y[0], original_y[-1]

    # Construct a step-based sequence that stays <= max_y, then explicitly add max_y if missing
    n_steps = int(np.floor((max_y - min_y) / step_size + 1e-12)) + 1
    base_y = min_y + step_size * np.arange(n_steps)

    # If max_y is not included due to rounding, append it to ensure the upper endpoint is present
    if not np.isclose(base_y[-1], max_y, rtol=0, atol=max(1e-12, step_size * 1e-9)):
        new_y_values = np.append(base_y, max_y)
    else:
        new_y_values = base_y

    # Interpolate X at the chosen Y grid
    new_x_values = np.interp(new_y_values, original_y, original_x)

    # Helper to append a point if its Y isn't already present (within tolerance)
    def append_point(xarr, yarr, xval, yval, tol):
        if not np.any(np.isclose(yarr, yval, rtol=0, atol=tol)):
            xarr = np.append(xarr, xval)
            yarr = np.append(yarr, yval)
        return xarr, yarr

    tol = max(1e-12, step_size * 1e-6)

    # Ensure the first and last input points are included exactly
    new_x_values, new_y_values = append_point(new_x_values, new_y_values, first_raw[0], first_raw[1], tol)
    new_x_values, new_y_values = append_point(new_x_values, new_y_values, last_raw[0], last_raw[1], tol)

    # Sort the combined set by Y to keep monotonic order
    order = np.argsort(new_y_values)
    new_y_values = new_y_values[order]
    new_x_values = new_x_values[order]

    return (original_x, original_y), (new_x_values, new_y_values), None


# --- 2. Build the Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title("📈 Curve Optimisation GUI")

st.sidebar.header("Controls")

default_data = (
    "0.0000,  0.0000\n  13.9578,  0.0007\n  17.2202,  0.0008\n  27.7589,  0.0014\n  35.1128,  0.0021\n  46.5671,  0.0027\n  48.4446,  0.0036\n  55.4269,  0.0040\n  70.0155,  0.0042\n  72.6182,  0.0050\n  84.2674,  0.0051\n  87.1151,  0.0055\n  91.9387,  0.0057\n  96.2453,  0.0059\n 103.9767,  0.0065\n 112.0285,  0.0067\n 125.6385,  0.0076\n 136.4149,  0.0083\n 143.8938,  0.0086\n 156.5457,  0.0090\n 161.1706,  0.0098\n 167.2981,  0.0099\n 171.9570,  0.0101\n 183.4499,  0.0103\n 186.5282,  0.0109\n 199.9672,  0.0117\n 200.6154,  0.0120\n 211.6856,  0.0122\n 221.5424,  0.0126\n 231.3057,  0.0132\n 236.9930,  0.0136\n 241.0742,  0.0139\n 245.2734,  0.0148\n 248.5471,  0.0152\n 263.4379,  0.0159\n 267.4800,  0.0165\n 268.0452,  0.0173\n 274.8515,  0.0176\n 281.3318,  0.0179\n 286.9734,  0.0182\n 288.1296,  0.0185\n 296.4628,  0.0186\n 305.8227,  0.0192\n 318.0909,  0.0193\n 330.4506,  0.0202\n 334.8660,  0.0209\n 342.8974,  0.0214\n 353.5063,  0.0217\n 359.5383,  0.0222\n 365.2831,  0.0229\n 379.0766,  0.0235\n 381.9534,  0.0240\n 393.9534,  0.0250\n 394.9998,  0.0253\n 404.8546,  0.0261\n 406.9277,  0.0267\n 414.7953,  0.0270\n 424.9932,  0.0274\n 435.2261,  0.0277\n 438.1643,  0.0287\n 449.7367,  0.0291\n 459.7187,  0.0292\n 470.6557,  0.0297\n 482.7122,  0.0300\n 493.8460,  0.0310\n 499.4568,  0.0314\n 506.9342,  0.0321\n 507.7926,  0.0327\n 516.4881,  0.0332\n 523.5039,  0.0337\n 538.0904,  0.0343\n 545.9434,  0.0350\n 554.2221,  0.0351\n 566.5399,  0.0355\n 571.0451,  0.0363\n 577.6600,  0.0372\n 580.8105,  0.0378\n 584.0170,  0.0385\n 590.8614,  0.0386\n 594.9391,  0.0395\n 603.0951,  0.0403\n 616.1698,  0.0406\n 617.9762,  0.0411\n 621.0663,  0.0419\n 629.4493,  0.0423\n 638.3280,  0.0428\n 642.7869,  0.0432\n 654.8551,  0.0435\n 667.0708,  0.0442\n 680.5015,  0.0447\n1363.0705,  0.1351\n1373.9931,  0.1356\n1379.8440,  0.1358\n1393.7250,  0.1363\n1408.3563,  0.1370\n1417.8119,  0.1374\n1418.5531,  0.1380\n1423.1332,  0.1385\n1435.4089,  0.1390\n1449.5135,  0.1399"
)

data_str = st.sidebar.text_area(
    "Input Data (X, Y per line)",
    value=default_data,
    height=250,
    help="Paste data here. Values can be separated by commas or spaces."
)

step_size = st.sidebar.number_input(
    "Step Size for Y-axis (Stress)",
    min_value=0.0001,
    value=0.005,
    step=0.001,
    format="%.4f"
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
        original_data, new_data, error = get_interpolated_data(data_str, step_size)
        if error:
            st.error(error)
            st.session_state.results_data = None
            st.session_state.just_generated = False
        else:
            st.session_state.original_data = original_data
            st.session_state.results_data = new_data
            st.session_state.generation_time = time.time()
            st.session_state.just_generated = True  # update text area on this rerun

# --- 5. Display the Output inside the placeholder if it Exists ---
if st.session_state.results_data is not None:
    with output_placeholder.container():
        generation_dt = datetime.fromtimestamp(st.session_state.generation_time)
        st.success(f"✅ Success! Curve generated at {generation_dt.strftime('%H:%M:%S')}.")

        col1, col2 = st.columns([1, 2])

        original_data = st.session_state.original_data
        new_data = st.session_state.results_data

        with col1:
            st.subheader("Interpolated Results")
            # Prepare CSV including retained endpoints
            result_lines = ["X (Strain),Y (Stress)"]
            for x, y in zip(new_data[0], new_data[1]):
                result_lines.append(f"{x:.2f},{y:.3f}")
            result_csv = "\n".join(result_lines)

            # Option B: overwrite widget state only right after generation
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
                file_name="interpolated_data.csv",
                mime="text/csv",
            )

        with col2:
            st.subheader("Plot")
            fig, ax = plt.subplots(figsize=(8, 6))
            # Input data (sorted by Y for visualization parity with interpolation)
            ax.plot(original_data[1], original_data[0], 'r-', marker='.', markersize=8, label='Input Data')
            # Interpolated data with endpoints retained
            ax.plot(new_data[1], new_data[0], 'bo', markersize=4, label='Interpolated Data')
            ax.set_xlabel('Y values (Stress)')
            ax.set_ylabel('X values (Strain)')
            ax.set_title('Input and Interpolated Data')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

        # Reset the flag so manual edits are preserved on later reruns
        st.session_state.just_generated = False
