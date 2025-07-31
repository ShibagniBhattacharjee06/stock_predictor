import gradio as gr
import numpy as np
import pandas as pd
import joblib

model = joblib.load("stock_price_model.joblib")
scaler = joblib.load("scaler.joblib")


def predict_next_close(open_price, high, low, close, volume, vwap):
    try:
        
        if any(val is None or val <= 0 for val in [open_price, high, low, close, volume, vwap]):
            return "❌ Please enter valid positive values for all fields"
        
        if high < max(open_price, close) or low > min(open_price, close):
            return "❌ Invalid price data: High should be ≥ max(Open, Close) and Low should be ≤ min(Open, Close)"
        
        
        import pandas as pd
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
        features_df = pd.DataFrame([[open_price, high, low, close, volume, vwap]], 
                                 columns=feature_names)
        
        scaled = scaler.transform(features_df)
        prediction = model.predict(scaled)
        
        # Calculate percentage change
        change_pct = ((prediction[0] - close) / close) * 100
        trend_emoji = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
        
        return f"{trend_emoji} ₹{prediction[0]:.2f} ({change_pct:+.2f}%)"
    
    except Exception as e:
        return f"❌ Error in prediction: {str(e)}"

# Custom CSS for modern dark theme
custom_css = """
/* Global Styles */
:root {
    --primary-color: #00d4aa;
    --secondary-color: #1e293b;
    --accent-color: #0f172a;
    --text-light: #e2e8f0;
    --success-color: #10b981;
    --error-color: #ef4444;
    --warning-color: #f59e0b;
}

body {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: var(--text-light);
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Main container */
.gradio-container {
    max-width: 1200px;
    margin: 0 auto;
    background: rgba(30, 41, 59, 0.3);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    border: 1px solid rgba(0, 212, 170, 0.2);
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
}

/* Header styling */
#title {
    text-align: center;
    background: linear-gradient(135deg, var(--primary-color), #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 2rem;
    text-shadow: 0 0 30px rgba(0, 212, 170, 0.3);
}

/* Input styling */
.gr-textbox, .gr-number {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 2px solid rgba(0, 212, 170, 0.3) !important;
    border-radius: 12px !important;
    color: var(--text-light) !important;
    transition: all 0.3s ease !important;
}

.gr-textbox:focus, .gr-number:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 20px rgba(0, 212, 170, 0.3) !important;
    transform: translateY(-2px) !important;
}

/* Button styling */
.gr-button {
    background: linear-gradient(135deg, var(--primary-color), #06b6d4) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 15px 30px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 25px rgba(0, 212, 170, 0.3) !important;
}

.gr-button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 15px 35px rgba(0, 212, 170, 0.4) !important;
    background: linear-gradient(135deg, #06b6d4, var(--primary-color)) !important;
}

/* Label styling */
label {
    color: var(--text-light) !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    margin-bottom: 8px !important;
}

/* Output box styling */
#output {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 2px solid rgba(0, 212, 170, 0.4) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    text-align: center !important;
    min-height: 60px !important;
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.3) !important;
}

/* Row styling */
.gr-row {
    gap: 20px !important;
    margin: 20px 0 !important;
}

/* Card-like sections */
.input-section {
    background: rgba(30, 41, 59, 0.5);
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    border: 1px solid rgba(0, 212, 170, 0.2);
    backdrop-filter: blur(10px);
}

/* Responsive design */
@media (max-width: 768px) {
    #title {
        font-size: 2rem;
    }
    
    .gr-button {
        font-size: 1rem !important;
        padding: 12px 24px !important;
    }
}

/* Animation keyframes */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.loading {
    animation: pulse 1.5s infinite;
}
"""

# Create the Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate"
    ),
    css=custom_css,
    title="Stock Price Predictor"
) as demo:
    
    # Header
    gr.HTML("""
        <div style="text-align: center; padding: 30px 0; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 15px; margin-bottom: 30px; border: 2px solid rgba(0, 212, 170, 0.3);">
            <h1 style="font-size: 3rem; font-weight: bold; margin: 0; background: linear-gradient(135deg, #00d4aa, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-shadow: 0 0 30px rgba(0, 212, 170, 0.3);">
                📈 AI Stock Price Predictor
            </h1>
            <h3 style="color: #e2e8f0; margin: 20px 0 10px 0; font-weight: 500; font-size: 1.3rem;">
                Predict tomorrow's closing price using advanced machine learning
            </h3>
            <p style="color: #94a3b8; font-style: italic; font-size: 1.1rem; margin: 0;">
                Enter today's market data to get intelligent price forecasts
            </p>
        </div>
    """)
    
    # Instructions
    with gr.Accordion("📋 How to Use", open=False):
        gr.Markdown("""
        **Step-by-step guide:**
        1. **Open Price**: The price at market opening
        2. **High Price**: The highest price during the trading day
        3. **Low Price**: The lowest price during the trading day  
        4. **Close Price**: The closing price of the current day
        5. **Volume**: Number of shares traded
        6. **10-day SMA**: Simple Moving Average over the last 10 days
        
        **Important Notes:**
        - All price values should be in INR (₹)
        - High price should be ≥ max(Open, Close)
        - Low price should be ≤ min(Open, Close)
        - Volume should be a positive number
        """)
    
    # Input sections
    with gr.Group():
        gr.HTML('<h3 style="color: #00d4aa; margin: 20px 0 15px 0; font-size: 1.4rem;">💰 Price Data</h3>')
        with gr.Row():
            open_input = gr.Number(
                label="🔓 Open Price (₹)",
                scale=1,
                minimum=0
            )
            high_input = gr.Number(
                label="📈 High Price (₹)",
                scale=1,
                minimum=0
            )
        
        with gr.Row():
            low_input = gr.Number(
                label="📉 Low Price (₹)",
                scale=1,
                minimum=0
            )
            close_input = gr.Number(
                label="🔒 Close Price (₹)",
                scale=1,
                minimum=0
            )
    
    with gr.Group():
        gr.HTML('<h3 style="color: #00d4aa; margin: 20px 0 15px 0; font-size: 1.4rem;">📊 Market Indicators</h3>')
        with gr.Row():
            volume_input = gr.Number(
                label="📦 Volume",
                scale=1,
                minimum=0
            )
            sma_input = gr.Number(
                label="📊 VWAP (₹)",
                scale=1,
                minimum=0
            )
    
    # Prediction section
    with gr.Group():
        predict_button = gr.Button(
            "🚀 Predict Tomorrow's Close Price",
            variant="primary",
            size="lg"
        )
        
        output = gr.Textbox(
            label="🎯 AI Prediction Result",
            value="Click predict to see tomorrow's forecasted closing price...",
            interactive=False,
            elem_id="output"
        )
    
    # Example data
    with gr.Accordion("💡 Try Example Data", open=False):
        gr.HTML("""
            <p style="color: #e2e8f0; margin-bottom: 15px;">
                Click the button below to load sample stock data:
            </p>
        """)
        
        def load_example():
            return 2750.50, 2780.25, 2735.00, 2765.80, 1250000, 2760.15
        
        example_button = gr.Button("📝 Load Example Values", variant="secondary")
        example_button.click(
            load_example,
            outputs=[open_input, high_input, low_input, close_input, volume_input, sma_input]
        )
    
    # Connect the prediction function
    predict_button.click(
        predict_next_close,
        inputs=[open_input, high_input, low_input, close_input, volume_input, sma_input],
        outputs=output
    )
    
    # Footer
    gr.HTML("""
        <div style="text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 2rem; padding: 20px; background: rgba(15, 23, 42, 0.3); border-radius: 10px; border: 1px solid rgba(100, 116, 139, 0.2);">
            ⚠️ <strong style="color: #f59e0b;">Disclaimer:</strong> This is a prediction model for educational purposes. 
            Always consult financial advisors before making investment decisions.
        </div>
    """)

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )