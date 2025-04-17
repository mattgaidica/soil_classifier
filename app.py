import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, interp1d
from matplotlib.gridspec import GridSpec

def find_intersection(x, y, target_percent):
    """Find the x-value where y equals the target percent."""
    f = interp1d(y, x, bounds_error=False, fill_value="extrapolate")
    return f(target_percent)

def create_plasticity_chart(ax):
    """Create the plasticity chart for fine-grained soils."""
    # Plot A-line and U-line
    ll = np.array([0, 100])
    a_line = 0.73 * (ll - 20)
    u_line = 0.9 * (ll - 8)
    
    ax.plot(ll, a_line, 'k-', label='A-line')
    ax.plot(ll, u_line, 'k--', label='U-line')
    
    # Add zones with larger font
    ax.text(25, 4, 'CL-ML', fontsize=10)
    ax.text(30, 15, 'CL', fontsize=10)
    ax.text(70, 40, 'CH', fontsize=10)
    ax.text(70, 15, 'MH', fontsize=10)
    ax.text(30, 5, 'ML', fontsize=10)
    
    # Set limits and labels with larger font
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_xlabel('Liquid Limit (LL)', fontsize=12)
    ax.set_ylabel('Plasticity Index (PI)', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Plasticity Chart', fontsize=14)
    ax.legend(fontsize=10)

def plot_grain_size_distribution(percent_passing, ax):
    """Create grain size distribution plot."""
    # Sieve sizes in mm (from largest to smallest)
    sieve_sizes = [25.4, 12.7, 9.5, 4.75, 2.0, 0.85, 0.425, 0.25, 
                   0.15, 0.075, 0.05, 0.02, 0.005, 0.002]
    
    # Filter out missing values
    valid_data = [(size, percent) for size, percent in zip(sieve_sizes, percent_passing) 
                  if percent is not None and percent != '']
    
    if not valid_data:
        raise ValueError("No valid data points provided")
    
    sizes = [point[0] for point in valid_data]
    percentages = [point[1] for point in valid_data]
    
    # Create smooth curve using linear interpolation
    x_smooth = np.logspace(np.log10(max(sizes)), np.log10(min(sizes)), 300)
    f = interp1d(percentages[::-1], sizes[::-1], bounds_error=False, fill_value="extrapolate")
    y_smooth = [f(p) for p in np.linspace(min(percentages), max(percentages), 300)]
    
    # Plot main curve
    ax.plot(x_smooth, y_smooth, 'b-', linewidth=1.5)
    ax.scatter(sizes, percentages, color='red', s=30, zorder=5)
    
    # Find and plot D60, D30, D10
    target_percents = [60, 30, 10]
    d_values = []
    
    # Add critical sieve lines first (behind the data)
    ax.axvline(x=4.75, color='darkred', linestyle='--', linewidth=1.5, alpha=0.8, 
               label='#4 Sieve (4.75mm)', zorder=1)
    ax.axvline(x=0.075, color='darkgreen', linestyle=':', linewidth=2, alpha=0.8, 
               label='#200 Sieve (0.075mm)', zorder=1)
    
    # Plot percent passing lines and points
    for target in target_percents:
        try:
            # Use linear interpolation for D-value calculations
            f = interp1d(percentages[::-1], sizes[::-1], bounds_error=False, fill_value="extrapolate")
            x_intersect = f(target)
            d_values.append(x_intersect)
            # Plot horizontal line in blue
            ax.axhline(y=target, color='blue', linestyle='-', linewidth=1, alpha=0.3)
            # Plot intersection point
            ax.scatter(x_intersect, target, color='blue', s=50, zorder=6)
            # Add label with larger font
            ax.annotate(f'D{target}={x_intersect:.3f}mm',
                       xy=(x_intersect, target),
                       xytext=(5, 5),
                       textcoords='offset points',
                       color='blue',
                       fontsize=10,
                       fontweight='bold')
        except:
            d_values.append(None)
    
    # Plot formatting with larger fonts
    ax.set_xscale('log')
    ax.set_xlim(100, 0.001)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_xlabel('Grain Size (mm)', fontsize=12)
    ax.set_ylabel('Percent Passing (%)', fontsize=12)
    ax.set_title('Particle Size Distribution', fontsize=14)
    ax.tick_params(labelsize=10)
    
    # Create larger legend with better positioning
    legend = ax.legend(fontsize=10, loc='lower left', 
                      bbox_to_anchor=(0.02, 0.02),
                      framealpha=0.9,
                      edgecolor='black')
    legend.get_frame().set_linewidth(0.5)
    
    return d_values

def determine_classification(percent_passing, d_values, liquid_limit=None, plasticity_index=None):
    """Determine USCS soil classification."""
    # Get percent passing #200 (0.075mm) and #4 (4.75mm)
    p200 = percent_passing[9]  # Index 9 is #200 sieve
    p4 = percent_passing[3]    # Index 3 is #4 sieve
    
    if p200 is None:
        return None, "Cannot classify: Missing #200 sieve data"
    
    classification = ""
    calc_text = []
    
    # Format the values with proper precision
    p200_str = f"{p200:.1f}%" if p200 is not None else "None"
    p4_str = f"{p4:.1f}%" if p4 is not None else "None"
    
    # Calculate Cu and Cc if possible
    cu_str = "None"
    cc_str = "None"
    if all(d_values):
        cu = d_values[0]/d_values[2]  # D60/D10
        cc = (d_values[1]**2)/(d_values[0]*d_values[2])  # (D30)²/(D60*D10)
        cu_str = f"{cu:.2f}"
        cc_str = f"{cc:.2f}"
    
    # Store the parameters for display
    calc_text.append(f"#200 passing: {p200_str}")
    calc_text.append(f"#4 passing: {p4_str}")
    if cu_str != "None":
        calc_text.append(f"Cu = {cu_str}, Cc = {cc_str}")
    
    # Handle NP values for classification
    is_non_plastic = (isinstance(liquid_limit, str) and liquid_limit.upper() == "NP") or \
                    (isinstance(plasticity_index, str) and plasticity_index.upper() == "NP")
    
    # Classification logic
    if p200 < 50:  # Coarse-grained
        coarse_retained = 100 - p4 if p4 is not None else None
        if coarse_retained is not None:
            if coarse_retained > 50:
                base = "G"  # Gravel
                calc_text.append("→ GRAVEL (>50% retained on #4)")
            else:
                base = "S"  # Sand
                calc_text.append("→ SAND (<50% retained on #4)")
            
            # Determine second letter based on fines and gradation
            if p200 < 5:
                if all(d_values):
                    if base == "G" and cu >= 4 and 1 <= cc <= 3:
                        classification = f"{base}W"
                        calc_text.append("→ Well-graded (Cu≥4, 1≤Cc≤3)")
                    elif base == "S" and cu >= 6 and 1 <= cc <= 3:
                        classification = f"{base}W"
                        calc_text.append("→ Well-graded (Cu≥6, 1≤Cc≤3)")
                    else:
                        classification = f"{base}P"
                        calc_text.append("→ Poorly-graded")
            elif p200 > 12:
                if is_non_plastic:
                    classification = f"{base}M"
                    calc_text.append("→ Silty fines (Non-plastic)")
                elif plasticity_index is not None and not isinstance(plasticity_index, str):
                    if plasticity_index > 7:
                        classification = f"{base}C"
                        calc_text.append("→ Clay fines (PI>7)")
                    else:
                        classification = f"{base}M"
                        calc_text.append("→ Silty fines (PI≤7)")
            else:  # 5% ≤ p200 ≤ 12%
                if is_non_plastic:
                    classification = f"{base}M"
                    calc_text.append("→ Silty fines (Non-plastic)")
                else:
                    # For dual classification, we need to consider both gradation and plasticity
                    if all(d_values):
                        well_graded = (base == "G" and cu >= 4 and 1 <= cc <= 3) or (base == "S" and cu >= 6 and 1 <= cc <= 3)
                        if well_graded:
                            if plasticity_index is not None:
                                if plasticity_index > 7:
                                    classification = f"{base}W-{base}C"
                                    calc_text.append("→ Well-graded with clay fines")
                                else:
                                    classification = f"{base}W-{base}M"
                                    calc_text.append("→ Well-graded with silty fines")
                            else:
                                classification = f"{base}W-{base}M"
                                calc_text.append("→ Well-graded with silty fines (default)")
                        else:
                            if plasticity_index is not None:
                                if plasticity_index > 7:
                                    classification = f"{base}P-{base}C"
                                    calc_text.append("→ Poorly-graded with clay fines")
                                else:
                                    classification = f"{base}P-{base}M"
                                    calc_text.append("→ Poorly-graded with silty fines")
                            else:
                                classification = f"{base}P-{base}M"
                                calc_text.append("→ Poorly-graded with silty fines (default)")
    else:  # Fine-grained
        if is_non_plastic:
            classification = "ML"
            calc_text.append("→ Silt (Non-plastic)")
        elif liquid_limit is not None and plasticity_index is not None and \
             not isinstance(liquid_limit, str) and not isinstance(plasticity_index, str):
            calc_text.append(f"LL = {liquid_limit}, PI = {plasticity_index}")
            if liquid_limit < 50:
                if plasticity_index < 4:
                    classification = "ML"
                    calc_text.append("→ Low plasticity silt (PI<4)")
                elif plasticity_index > 7:
                    classification = "CL"
                    calc_text.append("→ Low plasticity clay (PI>7)")
                else:
                    classification = "CL-ML"
                    calc_text.append("→ Silty clay (4≤PI≤7)")
            else:
                if plasticity_index > 0.73 * (liquid_limit - 20):
                    classification = "CH"
                    calc_text.append("→ High plasticity clay")
                else:
                    classification = "MH"
                    calc_text.append("→ High plasticity silt")
    
    # All possible classifications with current one highlighted
    all_classes = [
        "GW", "GP", "GM", "GC", "GW-GC", "GW-GM", "GP-GM", "GP-GC", "GM-GC",
        "SW", "SP", "SM", "SC", "SW-SC", "SW-SM", "SP-SM", "SP-SC", "SC-SM",
        "CL", "ML", "CH", "MH", "CL-ML"
    ]
    
    class_text = "\nPossible Classifications:\n"
    class_text += " ".join([f"[{c}]" if c == classification else c for c in all_classes])
    
    # Return the classification and calculation text
    return classification, "\n".join(calc_text) + "\n" + class_text

# Set page config
st.set_page_config(
    page_title="USCS Soil Classifier",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .classification-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .stTextInput>div>div>input {
        text-align: center;
    }
    .soil-class {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        padding: 0.5rem;
        background-color: #e6f3ff;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .classification-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px 0;
    }
    .classification-item {
        text-align: center;
        padding: 8px;
        border-radius: 4px;
    }
    .classification-selected {
        background-color: #ffeb3b;
        font-weight: bold;
    }
    .analysis-section {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin: 0;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #262730;
    }
    .decision-step {
        font-style: italic;
        margin: 0.25rem 0;
    }
    /* Remove extra spacing from markdown elements */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    .stMarkdown {
        margin: 0 !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("USCS Soil Classification Tool")
st.markdown("""
This tool helps classify soils according to the Unified Soil Classification System (USCS) according to ASTM D2487.
Enter the percent passing values for each sieve size and optional Atterberg limits.
""")

# Add soil data buttons in a single row
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

# Define soil data
soil_data = {
    'Soil 1': {
        'inch1': None, 'half_inch': None, 'three_eighth': None, 'no4': 100.0, 'no10': 82.0,
        'no20': 76.0, 'no40': 70.0, 'no60': 60.0, 'no100': 43.0, 'no200': 27.0,
        'p050': 23.0, 'p020': 13.0, 'p005': 8.0, 'p002': 3.0,
        'll': 18.0, 'pl': 12.0
    },
    'Soil 2': {
        'inch1': None, 'half_inch': 100.0, 'three_eighth': 98.0, 'no4': 95.0, 'no10': 93.0,
        'no20': 88.0, 'no40': 82.0, 'no60': 75.0, 'no100': 72.0, 'no200': 68.0,
        'p050': 66.0, 'p020': 33.0, 'p005': 21.0, 'p002': 10.0,
        'll': 56.0, 'pl': 31.0
    },
    'Soil 3': {
        'inch1': None, 'half_inch': None, 'three_eighth': None, 'no4': None, 'no10': None,
        'no20': None, 'no40': 100.0, 'no60': 98.0, 'no100': 96.0, 'no200': 95.0,
        'p050': 91.0, 'p020': 84.0, 'p005': 71.0, 'p002': 63.0,
        'll': 71.0, 'pl': 19.0
    },
    'Soil 4': {
        'inch1': None, 'half_inch': 75.0, 'three_eighth': 52.0, 'no4': 37.0, 'no10': 32.0,
        'no20': 23.0, 'no40': 11.0, 'no60': 7.0, 'no100': 4.0, 'no200': 2.0,
        'p050': None, 'p020': None, 'p005': None, 'p002': None,
        'll': "NP", 'pl': "NP"
    },
    'Soil 5': {
        'inch1': 100.0, 'half_inch': 87.0, 'three_eighth': 82.0, 'no4': 74.0, 'no10': 67.0,
        'no20': 48.0, 'no40': 36.0, 'no60': 22.0, 'no100': 16.0, 'no200': 13.0,
        'p050': 11.0, 'p020': 5.0, 'p005': 4.0, 'p002': 2.0,
        'll': 37.0, 'pl': 13.0
    },
    'Soil 6': {
        'inch1': None, 'half_inch': None, 'three_eighth': None, 'no4': None, 'no10': None,
        'no20': None, 'no40': None, 'no60': None, 'no100': 100.0, 'no200': 99.0,
        'p050': 82.0, 'p020': 37.0, 'p005': 8.0, 'p002': 6.0,
        'll': 20.0, 'pl': 15.0
    },
    'Soil 7': {
        'inch1': None, 'half_inch': None, 'three_eighth': None, 'no4': None, 'no10': None,
        'no20': 100.0, 'no40': 98.0, 'no60': 89.0, 'no100': 6.0, 'no200': 0.0,
        'p050': None, 'p020': None, 'p005': None, 'p002': None,
        'll': "NP", 'pl': "NP"
    },
    'Soil 8': {
        'inch1': 90.0, 'half_inch': 74.0, 'three_eighth': 69.0, 'no4': 64.0, 'no10': 51.0,
        'no20': 37.0, 'no40': 32.0, 'no60': 26.0, 'no100': 14.0, 'no200': 10.0,
        'p050': 10.0, 'p020': 8.0, 'p005': 5.0, 'p002': 2.0,
        'll': 43.0, 'pl': 24.0
    }
}

# Add buttons for each soil
with col1:
    if st.button("Soil 1", use_container_width=True):
        st.session_state.input_values = soil_data['Soil 1']
        st.session_state.classify_clicked = True
        st.rerun()
with col2:
    if st.button("Soil 2", use_container_width=True):
        st.session_state.input_values = soil_data['Soil 2']
        st.session_state.classify_clicked = True
        st.rerun()
with col3:
    if st.button("Soil 3", use_container_width=True):
        st.session_state.input_values = soil_data['Soil 3']
        st.session_state.classify_clicked = True
        st.rerun()
with col4:
    if st.button("Soil 4", use_container_width=True):
        st.session_state.input_values = soil_data['Soil 4']
        st.session_state.classify_clicked = True
        st.rerun()
with col5:
    if st.button("Soil 5", use_container_width=True):
        st.session_state.input_values = soil_data['Soil 5']
        st.session_state.classify_clicked = True
        st.rerun()
with col6:
    if st.button("Soil 6", use_container_width=True):
        st.session_state.input_values = soil_data['Soil 6']
        st.session_state.classify_clicked = True
        st.rerun()
with col7:
    if st.button("Soil 7", use_container_width=True):
        st.session_state.input_values = soil_data['Soil 7']
        st.session_state.classify_clicked = True
        st.rerun()
with col8:
    if st.button("Soil 8", use_container_width=True):
        st.session_state.input_values = soil_data['Soil 8']
        st.session_state.classify_clicked = True
        st.rerun()

# Initialize session state for input values if not exists
if 'input_values' not in st.session_state:
    st.session_state.input_values = {
        'inch1': None, 'half_inch': None, 'three_eighth': None, 'no4': None, 'no10': None,
        'no20': None, 'no40': None, 'no60': None, 'no100': None, 'no200': None,
        'p050': None, 'p020': None, 'p005': None, 'p002': None,
        'll': None, 'pl': None
    }

# Create three columns for input organization
col1, col2, col3 = st.columns(3)

# Sieve input fields
with col1:
    st.subheader("Coarse")
    input_labels = {
        'inch1': "1\" (25.4mm)",
        'half_inch': "1/2\" (12.7mm)",
        'three_eighth': "3/8\" (9.5mm)",
        'no4': "No. 4 (4.75mm)",
        'no10': "No. 10 (2.0mm)"
    }
    for key in ['inch1', 'half_inch', 'three_eighth', 'no4', 'no10']:
        value = st.text_input(
            input_labels[key],
            value=st.session_state.input_values[key] if st.session_state.input_values[key] is not None else "",
            key=f"input_{key}"
        )
        st.session_state.input_values[key] = float(value) if value.strip() else None

with col2:
    st.subheader("Fine")
    input_labels = {
        'no20': "No. 20 (0.85mm)",
        'no40': "No. 40 (0.425mm)",
        'no60': "No. 60 (0.25mm)",
        'no100': "No. 100 (0.15mm)",
        'no200': "No. 200 (0.075mm)"
    }
    for key in ['no20', 'no40', 'no60', 'no100', 'no200']:
        value = st.text_input(
            input_labels[key],
            value=st.session_state.input_values[key] if st.session_state.input_values[key] is not None else "",
            key=f"input_{key}"
        )
        st.session_state.input_values[key] = float(value) if value.strip() else None

with col3:
    st.subheader("Small & Limits")
    input_labels = {
        'p050': "0.050mm",
        'p020': "0.020mm",
        'p005': "0.005mm",
        'p002': "0.002mm",
        'll': "Liquid Limit (LL)",
        'pl': "Plastic Limit (PL)"
    }
    for key in ['p050', 'p020', 'p005', 'p002']:
        value = st.text_input(
            input_labels[key],
            value=st.session_state.input_values[key] if st.session_state.input_values[key] is not None else "",
            key=f"input_{key}"
        )
        st.session_state.input_values[key] = float(value) if value.strip() else None
    
    st.markdown("---")
    # Special handling for Atterberg limits
    for key in ['ll', 'pl']:
        value = st.text_input(
            input_labels[key],
            value=st.session_state.input_values[key] if st.session_state.input_values[key] is not None else "",
            key=f"input_{key}"
        )
        if value.strip():
            if value.upper() == "NP":
                st.session_state.input_values[key] = "NP"
            else:
                try:
                    st.session_state.input_values[key] = float(value)
                except ValueError:
                    st.error(f"Invalid value for {input_labels[key]}. Please enter a number or 'NP'.")

# Create a button to trigger classification
if st.button("Classify Soil", type="primary") or st.session_state.get('classify_clicked', False):
    # Reset the classify_clicked flag
    st.session_state.classify_clicked = False
    
    # Calculate PI if both LL and PL are numeric values
    pi = None
    if (st.session_state.input_values['ll'] is not None and 
        st.session_state.input_values['pl'] is not None):
        if (isinstance(st.session_state.input_values['ll'], str) or 
            isinstance(st.session_state.input_values['pl'], str)):
            pi = "NP"
        else:
            pi = st.session_state.input_values['ll'] - st.session_state.input_values['pl']
    
    # Collect all inputs into a list
    percent_passing = [
        st.session_state.input_values['inch1'],
        st.session_state.input_values['half_inch'],
        st.session_state.input_values['three_eighth'],
        st.session_state.input_values['no4'],
        st.session_state.input_values['no10'],
        st.session_state.input_values['no20'],
        st.session_state.input_values['no40'],
        st.session_state.input_values['no60'],
        st.session_state.input_values['no100'],
        st.session_state.input_values['no200'],
        st.session_state.input_values['p050'],
        st.session_state.input_values['p020'],
        st.session_state.input_values['p005'],
        st.session_state.input_values['p002']
    ]
    
    try:
        st.markdown("---")
        st.subheader("Classification Results")
        
        # Create grain size distribution plot
        st.markdown("### Grain Size Distribution")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), height_ratios=[2, 1])
        d_values = plot_grain_size_distribution(percent_passing, ax1)
        create_plasticity_chart(ax2)
        
        # Only plot on plasticity chart if values are numeric
        if (st.session_state.input_values['ll'] is not None and 
            pi is not None and 
            not isinstance(st.session_state.input_values['ll'], str) and 
            not isinstance(pi, str)):
            ax2.scatter(st.session_state.input_values['ll'], pi, color='red', s=50)
        
        st.pyplot(fig, use_container_width=True)
        
        # Display classification details
        st.markdown("### USCS Classification")
        
        # Get classification and calculation text
        classification, calc_text = determine_classification(
            percent_passing, d_values, st.session_state.input_values['ll'], pi)
        
        # Format the calculation text for better readability
        calc_lines = calc_text.split('\n')
        data_analysis = []
        possible_classifications = []
        determined_classification = None
        
        for line in calc_lines:
            if line.startswith("Data Analysis:"):
                continue
            elif line.startswith("Possible Classifications:"):
                break
            elif line.startswith("→"):
                data_analysis.append(line)
            else:
                data_analysis.append(line)
        
        # Extract key parameters
        p200_str = None
        p4_str = None
        cu_str = None
        cc_str = None
        
        for line in data_analysis:
            if "#200 passing:" in line:
                p200_str = line.split(":")[1].strip()
            elif "#4 passing:" in line:
                p4_str = line.split(":")[1].strip()
            elif "Cu =" in line:
                cu_str = line.split("Cu =")[1].split(",")[0].strip()
                cc_str = line.split("Cc =")[1].strip()
        
        # Create three columns for analysis sections
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        with analysis_col1:
            st.markdown(f'''
            <div class="analysis-section">
                <div class="section-title">Data Analysis</div>
                <table>
                    <thead>
                        <tr><th>Parameter</th><th>Value</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>#200 passing</td><td>{p200_str}</td></tr>
                        <tr><td>#4 passing</td><td>{p4_str}</td></tr>
                        <tr><td>Cu</td><td>{cu_str}</td></tr>
                        <tr><td>Cc</td><td>{cc_str}</td></tr>
                    </tbody>
                </table>
            </div>
            ''', unsafe_allow_html=True)
        
        with analysis_col2:
            decision_steps_html = '<div class="analysis-section"><div class="section-title">Decision Steps</div>'
            for line in data_analysis:
                if line.startswith("→"):
                    decision_steps_html += f'<div class="decision-step">{line}</div>'
            decision_steps_html += '</div>'
            st.markdown(decision_steps_html, unsafe_allow_html=True)
        
        with analysis_col3:
            table_html = """
            <table>
                <thead>
                    <tr><th>Parameter</th><th>Value</th></tr>
                </thead>
                <tbody>
                    <tr><td>Liquid Limit (LL)</td><td>{}</td></tr>
                    <tr><td>Plastic Limit (PL)</td><td>{}</td></tr>
                    <tr><td>Plasticity Index (PI)</td><td>{}</td></tr>
                </tbody>
            </table>
            """.format(
                "NP" if isinstance(st.session_state.input_values['ll'], str) else f"{st.session_state.input_values['ll']:.1f}",
                "NP" if isinstance(st.session_state.input_values['pl'], str) else f"{st.session_state.input_values['pl']:.1f}",
                "NP" if pi == "NP" else (f"{pi:.1f}" if pi is not None else "")
            ) if st.session_state.input_values['ll'] is not None else "<p>No Atterberg limits data available</p>"
            
            st.markdown(f'''
            <div class="analysis-section">
                <div class="section-title">Atterberg Limits</div>
                {table_html}
            </div>
            ''', unsafe_allow_html=True)
        
        # Possible Classifications section
        st.markdown("#### Possible Classifications")
        st.markdown(f"""
        <div class="classification-grid">
            <div class="classification-item{" classification-selected" if classification == "GW" else ""}">GW</div>
            <div class="classification-item{" classification-selected" if classification == "GP" else ""}">GP</div>
            <div class="classification-item{" classification-selected" if classification == "GM" else ""}">GM</div>
            <div class="classification-item{" classification-selected" if classification == "GC" else ""}">GC</div>
            <div class="classification-item{" classification-selected" if classification == "GW-GC" else ""}">GW-GC</div>
            <div class="classification-item{" classification-selected" if classification == "GW-GM" else ""}">GW-GM</div>
            <div class="classification-item{" classification-selected" if classification == "GP-GM" else ""}">GP-GM</div>
            <div class="classification-item{" classification-selected" if classification == "GP-GC" else ""}">GP-GC</div>
            <div class="classification-item{" classification-selected" if classification == "GM-GC" else ""}">GM-GC</div>
            <div class="classification-item{" classification-selected" if classification == "SW" else ""}">SW</div>
            <div class="classification-item{" classification-selected" if classification == "SP" else ""}">SP</div>
            <div class="classification-item{" classification-selected" if classification == "SM" else ""}">SM</div>
            <div class="classification-item{" classification-selected" if classification == "SC" else ""}">SC</div>
            <div class="classification-item{" classification-selected" if classification == "SW-SC" else ""}">SW-SC</div>
            <div class="classification-item{" classification-selected" if classification == "SW-SM" else ""}">SW-SM</div>
            <div class="classification-item{" classification-selected" if classification == "SP-SM" else ""}">SP-SM</div>
            <div class="classification-item{" classification-selected" if classification == "SP-SC" else ""}">SP-SC</div>
            <div class="classification-item{" classification-selected" if classification == "SC-SM" else ""}">SC-SM</div>
            <div class="classification-item{" classification-selected" if classification == "CL" else ""}">CL</div>
            <div class="classification-item{" classification-selected" if classification == "ML" else ""}">ML</div>
            <div class="classification-item{" classification-selected" if classification == "CH" else ""}">CH</div>
            <div class="classification-item{" classification-selected" if classification == "MH" else ""}">MH</div>
            <div class="classification-item{" classification-selected" if classification == "CL-ML" else ""}">CL-ML</div>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error in classification: {str(e)}") 