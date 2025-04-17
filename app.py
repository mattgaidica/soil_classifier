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
    
    # Add zones with smaller font
    ax.text(25, 4, 'CL-ML', fontsize=6)
    ax.text(30, 15, 'CL', fontsize=6)
    ax.text(70, 40, 'CH', fontsize=6)
    ax.text(70, 15, 'MH', fontsize=6)
    ax.text(30, 5, 'ML', fontsize=6)
    
    # Set limits and labels with smaller font
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_xlabel('Liquid Limit (LL)', fontsize=8)
    ax.set_ylabel('Plasticity Index (PI)', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_title('Plasticity Chart', fontsize=9)

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
    
    # Create smooth curve
    x_smooth = np.logspace(np.log10(max(sizes)), np.log10(min(sizes)), 300)
    spl = make_interp_spline(sizes[::-1], percentages[::-1], k=3)
    y_smooth = spl(x_smooth)
    
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
            x_intersect = find_intersection(x_smooth, y_smooth, target)
            d_values.append(x_intersect)
            # Plot horizontal line in blue
            ax.axhline(y=target, color='blue', linestyle='-', linewidth=1, alpha=0.3)
            # Plot intersection point
            ax.scatter(x_intersect, target, color='blue', s=50, zorder=6)
            # Add label
            ax.annotate(f'D{target}={x_intersect:.3f}mm',
                       xy=(x_intersect, target),
                       xytext=(5, 5),
                       textcoords='offset points',
                       color='blue',
                       fontsize=7,
                       fontweight='bold')
        except:
            d_values.append(None)
    
    # Plot formatting
    ax.set_xscale('log')
    ax.set_xlim(100, 0.001)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_xlabel('Grain Size (mm)', fontsize=8)
    ax.set_ylabel('Percent Passing (%)', fontsize=8)
    ax.set_title('Particle Size Distribution', fontsize=9)
    ax.tick_params(labelsize=6)
    
    # Create larger legend with better positioning
    legend = ax.legend(fontsize=8, loc='lower left', 
                      bbox_to_anchor=(0.02, 0.02),
                      framealpha=0.9,
                      edgecolor='black')
    legend.get_frame().set_linewidth(0.5)
    
    return d_values

def determine_classification(percent_passing, d_values, liquid_limit=None, plasticity_index=None):
    """Determine USCS soil classification."""
    # Get percent passing #200 (0.075mm)
    p200 = None
    for size, percent in zip([0.075], [percent_passing[9]]):  # Index 9 is #200 sieve
        if percent is not None:
            p200 = percent
    
    if p200 is None:
        return None, "Cannot classify: Missing #200 sieve data"
    
    # Get percent passing #4 (4.75mm)
    p4 = None
    for size, percent in zip([4.75], [percent_passing[3]]):  # Index 3 is #4 sieve
        if percent is not None:
            p4 = percent
    
    classification = ""
    calc_text = f"Data Analysis:\n"
    calc_text += f"#200 passing: {p200:.1f}%\n"
    
    if p200 < 50:  # Coarse-grained
        calc_text += f"#4 passing: {p4:.1f}%\n"
        
        # Calculate Cu and Cc if possible
        if all(d_values):
            cu = d_values[0]/d_values[2]  # D60/D10
            cc = (d_values[1]**2)/(d_values[0]*d_values[2])  # (D30)²/(D60*D10)
            calc_text += f"Cu = {cu:.2f}, Cc = {cc:.2f}\n"
        
        # Determine if Gravel or Sand
        coarse_retained = 100 - p4 if p4 is not None else None
        if coarse_retained is not None:
            if coarse_retained > 50:
                base = "G"  # Gravel
                calc_text += "→ GRAVEL (>50% retained on #4)\n"
            else:
                base = "S"  # Sand
                calc_text += "→ SAND (<50% retained on #4)\n"
            
            # Determine second letter based on fines and gradation
            if p200 < 5:
                if all(d_values):
                    if base == "G" and cu >= 4 and 1 <= cc <= 3:
                        classification = f"{base}W"
                        calc_text += "→ Well-graded (Cu≥4, 1≤Cc≤3)\n"
                    elif base == "S" and cu >= 6 and 1 <= cc <= 3:
                        classification = f"{base}W"
                        calc_text += "→ Well-graded (Cu≥6, 1≤Cc≤3)\n"
                    else:
                        classification = f"{base}P"
                        calc_text += "→ Poorly-graded\n"
            elif p200 > 12:
                if plasticity_index is not None:
                    if plasticity_index > 7:
                        classification = f"{base}C"
                        calc_text += "→ Clay fines (PI>7)\n"
                    else:
                        classification = f"{base}M"
                        calc_text += "→ Silty fines (PI≤7)\n"
            else:
                classification = f"{base}P-{base}M"  # Dual classification
                calc_text += "→ Dual classification (5%<fines<12%)\n"
    else:  # Fine-grained
        if liquid_limit is not None and plasticity_index is not None:
            calc_text += f"LL = {liquid_limit}, PI = {plasticity_index}\n"
            if liquid_limit < 50:
                if plasticity_index < 4:
                    classification = "ML"
                    calc_text += "→ Low plasticity silt (PI<4)\n"
                elif plasticity_index > 7:
                    classification = "CL"
                    calc_text += "→ Low plasticity clay (PI>7)\n"
                else:
                    classification = "CL-ML"
                    calc_text += "→ Silty clay (4≤PI≤7)\n"
            else:
                if plasticity_index > 0.73 * (liquid_limit - 20):
                    classification = "CH"
                    calc_text += "→ High plasticity clay\n"
                else:
                    classification = "MH"
                    calc_text += "→ High plasticity silt\n"
    
    # All possible classifications with current one highlighted
    all_classes = ["GW", "GP", "GM", "GC", "GW-GM", "GW-GC", "GP-GM", "GP-GC", 
                  "SW", "SP", "SM", "SC", "SW-SM", "SW-SC", "SP-SM", "SP-SC",
                  "ML", "CL", "MH", "CH", "CL-ML"]
    
    class_text = "\nPossible Classifications:\n"
    class_text += " ".join([f"[{c}]" if c == classification else c for c in all_classes])
    
    return classification, calc_text + class_text

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
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌍 USCS Soil Classification Tool")
st.markdown("""
This tool helps classify soils according to the Unified Soil Classification System (USCS).
Enter the percent passing values for each sieve size and optional Atterberg limits.
""")

# Create three columns for input organization
col1, col2, col3 = st.columns(3)

# Initialize session state for input values if not exists
if 'input_values' not in st.session_state:
    st.session_state.input_values = {
        'inch1': None, 'half_inch': None, 'three_eighth': None, 'no4': None, 'no10': None,
        'no20': None, 'no40': None, 'no60': None, 'no100': None, 'no200': None,
        'p050': None, 'p020': None, 'p005': None, 'p002': None,
        'll': None, 'pl': None
    }

# Sieve input fields
with col1:
    st.subheader("Coarse Sieves")
    st.session_state.input_values['inch1'] = st.number_input("1\" (25.4mm)", min_value=0.0, max_value=100.0, 
                                                            value=st.session_state.input_values['inch1'], step=0.1, format="%.1f")
    st.session_state.input_values['half_inch'] = st.number_input("1/2\" (12.7mm)", min_value=0.0, max_value=100.0, 
                                                                value=st.session_state.input_values['half_inch'], step=0.1, format="%.1f")
    st.session_state.input_values['three_eighth'] = st.number_input("3/8\" (9.5mm)", min_value=0.0, max_value=100.0, 
                                                                   value=st.session_state.input_values['three_eighth'], step=0.1, format="%.1f")
    st.session_state.input_values['no4'] = st.number_input("No. 4 (4.75mm)", min_value=0.0, max_value=100.0, 
                                                          value=st.session_state.input_values['no4'], step=0.1, format="%.1f")
    st.session_state.input_values['no10'] = st.number_input("No. 10 (2.0mm)", min_value=0.0, max_value=100.0, 
                                                           value=st.session_state.input_values['no10'], step=0.1, format="%.1f")

with col2:
    st.subheader("Fine Sieves")
    st.session_state.input_values['no20'] = st.number_input("No. 20 (0.85mm)", min_value=0.0, max_value=100.0, 
                                                           value=st.session_state.input_values['no20'], step=0.1, format="%.1f")
    st.session_state.input_values['no40'] = st.number_input("No. 40 (0.425mm)", min_value=0.0, max_value=100.0, 
                                                           value=st.session_state.input_values['no40'], step=0.1, format="%.1f")
    st.session_state.input_values['no60'] = st.number_input("No. 60 (0.25mm)", min_value=0.0, max_value=100.0, 
                                                           value=st.session_state.input_values['no60'], step=0.1, format="%.1f")
    st.session_state.input_values['no100'] = st.number_input("No. 100 (0.15mm)", min_value=0.0, max_value=100.0, 
                                                            value=st.session_state.input_values['no100'], step=0.1, format="%.1f")
    st.session_state.input_values['no200'] = st.number_input("No. 200 (0.075mm)", min_value=0.0, max_value=100.0, 
                                                            value=st.session_state.input_values['no200'], step=0.1, format="%.1f")

with col3:
    st.subheader("Small Particles & Limits")
    st.session_state.input_values['p050'] = st.number_input("0.050mm", min_value=0.0, max_value=100.0, 
                                                           value=st.session_state.input_values['p050'], step=0.1, format="%.1f")
    st.session_state.input_values['p020'] = st.number_input("0.020mm", min_value=0.0, max_value=100.0, 
                                                           value=st.session_state.input_values['p020'], step=0.1, format="%.1f")
    st.session_state.input_values['p005'] = st.number_input("0.005mm", min_value=0.0, max_value=100.0, 
                                                           value=st.session_state.input_values['p005'], step=0.1, format="%.1f")
    st.session_state.input_values['p002'] = st.number_input("0.002mm", min_value=0.0, max_value=100.0, 
                                                           value=st.session_state.input_values['p002'], step=0.1, format="%.1f")
    
    st.markdown("---")
    st.session_state.input_values['ll'] = st.number_input("Liquid Limit (LL)", min_value=0.0, max_value=200.0, 
                                                         value=st.session_state.input_values['ll'], step=0.1, format="%.1f")
    st.session_state.input_values['pl'] = st.number_input("Plastic Limit (PL)", min_value=0.0, max_value=200.0, 
                                                         value=st.session_state.input_values['pl'], step=0.1, format="%.1f")

# Add a sample data button
if st.button("Load Sample Data (Soil 1)", type="secondary"):
    # Sample data for Soil 1
    sample_data = {
        'inch1': None, 'half_inch': None, 'three_eighth': None, 'no4': 100.0, 'no10': 82.0,
        'no20': 76.0, 'no40': 70.0, 'no60': 60.0, 'no100': 43.0, 'no200': 27.0,
        'p050': 23.0, 'p020': 13.0, 'p005': 8.0, 'p002': 3.0,
        'll': 35.0, 'pl': 15.0
    }
    st.session_state.input_values = sample_data
    st.rerun()

# Create a button to trigger classification
if st.button("Classify Soil", type="primary"):
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
    
    # Calculate PI if both LL and PL are provided
    pi = None
    if st.session_state.input_values['ll'] is not None and st.session_state.input_values['pl'] is not None:
        pi = st.session_state.input_values['ll'] - st.session_state.input_values['pl']
    
    try:
        st.markdown("---")
        st.subheader("Classification Results")
        
        # Create two columns for results
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            # Create grain size distribution plot
            st.markdown("### Grain Size Distribution")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[2, 1])
            d_values = plot_grain_size_distribution(percent_passing, ax1)
            create_plasticity_chart(ax2)
            if st.session_state.input_values['ll'] is not None and pi is not None:
                ax2.scatter(st.session_state.input_values['ll'], pi, color='red', s=50)
            st.pyplot(fig)
        
        # Display classification details in its own row
        st.markdown("---")
        st.markdown("### Classification Details")
        classification, calc_text = determine_classification(
            percent_passing, d_values, st.session_state.input_values['ll'], pi)
        
        with st.container(border=True):
            st.markdown("#### Classification Process")
            st.markdown(calc_text)
            
            if st.session_state.input_values['ll'] is not None and st.session_state.input_values['pl'] is not None:
                st.markdown(f"""
                #### Atterberg Limits
                - Liquid Limit (LL) = {st.session_state.input_values['ll']}
                - Plastic Limit (PL) = {st.session_state.input_values['pl']}
                - Plasticity Index (PI) = {pi}
                """)
        
        # Display classification in its own row at the bottom
        st.markdown("---")
        st.markdown("### USCS Classification")
        if classification:
            st.markdown(f'<div class="soil-class">{classification}</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error in classification: {str(e)}")

# Add footer with information
st.markdown("---")
st.markdown("""
💡 **Note:** This tool implements the Unified Soil Classification System (USCS) according to ASTM D2487.
""") 