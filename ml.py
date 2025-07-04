# -*- coding: utf-8 -*-
"""ml.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nN6-F3s1jqYkYdMwkmOUAqD9Yfc7bjUv
"""


# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.stats import gaussian_kde
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("husl")

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'features' not in st.session_state:
    st.session_state.features = []
if 'target' not in st.session_state:
    st.session_state.target = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = "Random Forest"
if 'test_size' not in st.session_state:
    st.session_state.test_size = 20

# Custom CSS for styling
st.markdown("""
<style>
:root {
    --primary: #6366f1;
    --secondary: #8b5cf6;
    --accent: #ec4899;
    --dark: #1e293b;
    --light: #f8fafc;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: var(--light);
}

[data-testid="stHeader"] {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(10px);
}

[data-testid="stSidebar"] {
    background: rgba(30, 41, 59, 0.8) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
}

.st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj {
    border-color: rgba(255, 255, 255, 0.1) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(30, 41, 59, 0.5) !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    margin: 0 !important;
    color: #cbd5e1 !important;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(to right, var(--primary), var(--secondary)) !important;
    color: white !important;
    font-weight: bold;
}

.stButton button {
    background: linear-gradient(to right, var(--primary), var(--secondary)) !important;
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4);
}

.stButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6);
}

.stSlider .st-c7 {
    background: linear-gradient(to right, var(--primary), var(--secondary)) !important;
}

.stSelectbox, .stMultiselect {
    background: rgba(30, 41, 59, 0.7) !important;
    border-radius: 12px !important;
    padding: 10px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.stDataFrame {
    border-radius: 15px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    background: linear-gradient(to right, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card {
    background: rgba(30, 41, 59, 0.7) !important;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.success-card {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}

.info-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.warning-card {
    background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
}

</style>
""", unsafe_allow_html=True)

def load_dataset(file):
    """Load and preview dataset"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            st.error("❌ Please upload a CSV or Excel file")
            return None

        st.session_state.dataset = df
        return df

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def get_dataset_info(df):
    """Get detailed dataset information"""
    if df is None:
        return

    info_html = f"""
    <div class="info-card">
        <h3>📈 Dataset Information</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                <h4>📊 Basic Stats</h4>
                <p><strong>Rows:</strong> {df.shape[0]}</p>
                <p><strong>Columns:</strong> {df.shape[1]}</p>
                <p><strong>Memory:</strong> {df.memory_usage(deep=True).sum() / 1024:.2f} KB</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                <h4>🔍 Data Types</h4>
                <p><strong>Numeric:</strong> {df.select_dtypes(include=[np.number]).shape[1]}</p>
                <p><strong>Categorical:</strong> {df.select_dtypes(include=['object']).shape[1]}</p>
                <p><strong>Missing Values:</strong> {df.isnull().sum().sum()}</p>
            </div>
        </div>
    </div>
    """
    st.markdown(info_html, unsafe_allow_html=True)

def train_model():
    """Train the selected model"""
    if st.session_state.dataset is None:
        st.warning("⚠️ Please upload a dataset first")
        return

    if not st.session_state.features or not st.session_state.target:
        st.warning("⚠️ Please select features and target")
        return

    try:
        # Prepare data
        df = st.session_state.dataset
        X = df[st.session_state.features].copy()
        y = df[st.session_state.target].copy()

        # Handle categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                st.session_state.label_encoders[col] = le

        # Handle target variable
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
            st.session_state.label_encoders['target'] = le_target
            task_type = 'classification'
        else:
            task_type = 'regression' if len(y.unique()) > 10 else 'classification'

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=st.session_state.test_size/100,
            random_state=42, stratify=y if task_type == 'classification' else None
        )

        # Store in session state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        # Scale features
        X_train_scaled = st.session_state.scaler.fit_transform(X_train)
        X_test_scaled = st.session_state.scaler.transform(X_test)

        # Model selection
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42) if task_type == 'classification' else RandomForestRegressor(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42) if task_type == 'classification' else LinearRegression(),
            'SVM': SVC(random_state=42) if task_type == 'classification' else None,
            'Decision Tree': DecisionTreeClassifier(random_state=42) if task_type == 'classification' else None,
            'K-Nearest Neighbors': KNeighborsClassifier() if task_type == 'classification' else None,
            'Naive Bayes': GaussianNB() if task_type == 'classification' else None,
            'Gradient Boosting': GradientBoostingClassifier(random_state=42) if task_type == 'classification' else None,
        }

        if st.session_state.model_name not in models or models[st.session_state.model_name] is None:
            st.warning(f"⚠️ {st.session_state.model_name} not available for {task_type}")
            return

        model = models[st.session_state.model_name]

        # Train model
        if st.session_state.model_name in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Store trained model
        st.session_state.trained_model = model

        # Calculate metrics
        if task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            result_html = f"""
            <div class="success-card">
                <h3>🎯 Model Training Complete!</h3>
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px; margin-top: 15px;">
                    <h4>📊 Results</h4>
                    <p><strong>Model:</strong> {st.session_state.model_name}</p>
                    <p><strong>Accuracy:</strong> {accuracy:.4f} ({accuracy*100:.2f}%)</p>
                    <p><strong>Task Type:</strong> {task_type.title()}</p>
                    <p><strong>Features Used:</strong> {len(st.session_state.features)}</p>
                    <p><strong>Training Samples:</strong> {len(X_train)}</p>
                    <p><strong>Test Samples:</strong> {len(X_test)}</p>
                </div>
            </div>
            """
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            result_html = f"""
            <div class="success-card">
                <h3>🎯 Model Training Complete!</h3>
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px; margin-top: 15px;">
                    <h4>📊 Results</h4>
                    <p><strong>Model:</strong> {st.session_state.model_name}</p>
                    <p><strong>R² Score:</strong> {r2:.4f}</p>
                    <p><strong>MSE:</strong> {mse:.4f}</p>
                    <p><strong>RMSE:</strong> {np.sqrt(mse):.4f}</p>
                    <p><strong>Task Type:</strong> {task_type.title()}</p>
                    <p><strong>Features Used:</strong> {len(st.session_state.features)}</p>
                </div>
            </div>
            """

        st.markdown(result_html, unsafe_allow_html=True)
        st.success(f"✅ Model trained successfully with {accuracy*100:.2f}% accuracy!" if task_type == 'classification' else f"✅ Model trained successfully with R² = {r2:.4f}!")

    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")

def create_confusion_matrix():
    """Create confusion matrix visualization"""
    if st.session_state.trained_model is None or st.session_state.X_test is None:
        st.warning("⚠️ Please train a model first")
        return None

    try:
        # Get predictions
        model = st.session_state.trained_model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        if model.__class__.__name__ in ['SVC', 'LogisticRegression', 'KNeighborsClassifier', 'GaussianNB']:
            y_pred = model.predict(st.session_state.scaler.transform(X_test))
        else:
            y_pred = model.predict(X_test)

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Create beautiful heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title('🎯 Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Labels', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Labels', fontsize=12, fontweight='bold')
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error creating confusion matrix: {str(e)}")

def create_feature_importance():
    """Create feature importance visualization"""
    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first")
        return

    try:
        model = st.session_state.trained_model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = st.session_state.features

            # Create beautiful bar plot
            fig, ax = plt.subplots(figsize=(12, 8))
            indices = np.argsort(importances)[::-1]

            colors = plt.cm.viridis(np.linspace(0, 1, len(importances)))
            bars = ax.bar(range(len(importances)), importances[indices], color=colors)

            ax.set_title('🔍 Feature Importance', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Features', fontweight='bold')
            ax.set_ylabel('Importance', fontweight='bold')
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right', fontsize=10)

            # Add value labels on bars
            for i, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{importances[indices[i]]:.3f}', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("⚠️ This model doesn't provide feature importance")

    except Exception as e:
        st.error(f"Error creating feature importance plot: {str(e)}")

def create_model_performance():
    """Create model performance visualization"""
    if st.session_state.trained_model is None or st.session_state.X_test is None:
        st.warning("⚠️ Please train a model first")
        return

    try:
        # Get predictions
        model = st.session_state.trained_model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        if model.__class__.__name__ in ['SVC', 'LogisticRegression', 'KNeighborsClassifier', 'GaussianNB']:
            y_pred = model.predict(st.session_state.scaler.transform(X_test))
        else:
            y_pred = model.predict(X_test)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📊 Model Performance Dashboard', fontsize=16, fontweight='bold')

        # 1. Prediction vs Actual
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='royalblue', edgecolors='white', linewidth=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values', fontweight='bold')
        axes[0, 0].set_ylabel('Predicted Values', fontweight='bold')
        axes[0, 0].set_title('🎯 Predictions vs Actual', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuals
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green', edgecolors='white', linewidth=0.5)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Predicted Values', fontweight='bold')
        axes[0, 1].set_ylabel('Residuals', fontweight='bold')
        axes[0, 1].set_title('📈 Residual Plot', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Distribution of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='white', linewidth=0.5)
        axes[1, 0].set_xlabel('Residuals', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('📊 Residual Distribution', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('📉 Q-Q Plot', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error creating performance plots: {str(e)}")

def create_data_distribution():
    """Create data distribution visualization"""
    if st.session_state.dataset is None:
        st.warning("⚠️ Please upload a dataset first")
        return

    try:
        df = st.session_state.dataset
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            st.warning("⚠️ No numeric columns found for distribution plot")
            return

        # Create subplots
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle('📊 Data Distribution Dashboard', fontsize=16, fontweight='bold')

        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                # Create histogram with KDE
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='white', density=True)

                # Add KDE curve
                data = df[col].dropna()
                if len(data) > 1:
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 100)
                    axes[i].plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')

                axes[i].set_xlabel(col, fontweight='bold')
                axes[i].set_ylabel('Density', fontweight='bold')
                axes[i].set_title(f'📈 {col} Distribution', fontweight='bold')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()

        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error creating distribution plots: {str(e)}")

# Main App
def main():
    st.set_page_config(
        page_title="🤖 AI/ML Model Training Studio",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: white; font-size: 1.8em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🤖 AI/ML Studio
            </h1>
            <p style="color: #cbd5e1; font-size: 1em; margin: 5px 0 0 0; opacity: 0.9;">
                No-code Machine Learning
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        uploaded_file = st.file_uploader(
            "📂 Upload Dataset (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your dataset to get started"
        )

        if uploaded_file:
            df = load_dataset(uploaded_file)
            if df is not None:
                st.session_state.dataset = df
                st.success(f"✅ Dataset loaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")

                # Show dataset info
                get_dataset_info(df)

                # Feature and target selection
                st.session_state.features = st.multiselect(
                    "🎲 Select Features",
                    options=df.columns.tolist(),
                    default=st.session_state.features if st.session_state.features else [],
                    help="Choose input features for training"
                )

                st.session_state.target = st.selectbox(
                    "🎯 Select Target",
                    options=df.columns.tolist(),
                    index=df.columns.tolist().index(st.session_state.target) if st.session_state.target else 0,
                    help="Choose the target variable to predict"
                )

                st.session_state.model_name = st.selectbox(
                    "🤖 Select Model",
                    options=[
                        "Random Forest",
                        "Logistic Regression",
                        "SVM",
                        "Decision Tree",
                        "K-Nearest Neighbors",
                        "Naive Bayes",
                        "Gradient Boosting"
                    ],
                    index=0,
                    help="Choose the machine learning algorithm"
                )

                st.session_state.test_size = st.slider(
                    "📊 Test Size (%)",
                    min_value=10,
                    max_value=40,
                    value=20,
                    step=5,
                    help="Percentage of data for testing"
                )

                if st.button("🚀 Train Model", use_container_width=True):
                    with st.spinner('Training model...'):
                        train_model()

    # Main Content
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px; border-radius: 15px; margin-bottom: 20px;">
        <h1 style="color: white; font-size: 2.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🤖 AI/ML Model Training Studio
        </h1>
        <p style="color: white; font-size: 1.2em; margin: 10px 0 0 0; opacity: 0.9;">
            Upload your dataset, select features, train models, and visualize results with beautiful charts!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dataset Preview", "🎯 Model Training", "📈 Visualizations"])

    with tab1:
        if st.session_state.dataset is not None:
            st.subheader("Dataset Preview")
            st.dataframe(st.session_state.dataset.head(10), use_container_width=True)
        else:
            st.info("ℹ️ Upload a dataset to get started", icon="ℹ️")

    with tab2:
        if st.session_state.trained_model:
            st.subheader("Model Evaluation")

            # Get predictions for classification report
            model = st.session_state.trained_model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            if model.__class__.__name__ in ['SVC', 'LogisticRegression', 'KNeighborsClassifier', 'GaussianNB']:
                y_pred = model.predict(st.session_state.scaler.transform(X_test))
            else:
                y_pred = model.predict(X_test)

            # Classification report
            st.subheader("📝 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)

    with tab3:
        st.subheader("Model Visualization Tools")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🎯 Confusion Matrix", use_container_width=True):
                create_confusion_matrix()
        with col2:
            if st.button("⭐ Feature Importance", use_container_width=True):
                create_feature_importance()
        with col3:
            if st.button("📈 Model Performance", use_container_width=True):
                create_model_performance()
        with col4:
            if st.button("📊 Data Distribution", use_container_width=True):
                create_data_distribution()

        if st.session_state.dataset is None:
            st.info("ℹ️ Upload a dataset and train a model to see visualizations", icon="ℹ️")

# Run the app
if __name__ == "__main__":
    main()
