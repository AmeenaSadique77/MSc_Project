from flask import Flask, request, render_template
import pandas as pd
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load your datasets when the application starts
phenotypes_df = pd.read_excel('Combined_Phenotypes_Final.xlsx')
genes_df = pd.read_excel('Final_Extended_Combined_Gene_CDS.xlsx')

# Load the trained model, scaler, and encoders
model = tf.keras.models.load_model('final_model.keras')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

# Filling NaNs with the median value for numeric columns
phenotypes_df_filled = phenotypes_df.fillna(phenotypes_df.median(numeric_only=True))
genes_df_filled = genes_df.fillna(genes_df.median(numeric_only=True))

# Filling remaining NaNs with 'Unknown'
phenotypes_df_filled = phenotypes_df_filled.fillna('Unknown')
genes_df_filled = genes_df_filled.fillna('Unknown')

# Verify that all NaNs have been filled
phenotypes_nan_check = phenotypes_df_filled.isna().sum()
genes_nan_check = genes_df_filled.isna().sum()

print(phenotypes_nan_check)
print(genes_nan_check)

# Combine the datasets on the 'Gene' column
combined_data = pd.merge(genes_df, phenotypes_df, on='Gene')

def predict_with_gene_phenotype(gene, phenotype, combined_data, model, scaler, label_encoders, target_encoder):
    # Filter the row corresponding to the input gene and phenotype
    gene_phenotype_row = combined_data[(combined_data['Gene'] == gene) & (combined_data['Phenotype_y'] == phenotype)]

    if gene_phenotype_row.empty:
        raise ValueError("Gene and Phenotype combination not found in the dataset.")

    # Drop the target column
    gene_features = gene_phenotype_row.drop(columns=['Phenotype_x'])

    # Fill NaN values before encoding
    gene_features_filled = gene_features.fillna('Unknown')

    # Encode categorical features
    for column in gene_features_filled.select_dtypes(include=['object']).columns:
        if column in label_encoders:
            le = label_encoders[column]
            gene_features_filled[column] = le.transform(gene_features_filled[column])

    # Scale the features
    gene_features_scaled = scaler.transform(gene_features_filled)

    # Make predictions
    predictions = model.predict(gene_features_scaled)
    predicted_class = tf.argmax(predictions, axis=1)

    # Decode the predicted class
    predicted_phenotype = target_encoder.inverse_transform(predicted_class)

    # Create a dictionary for the output with additional information
    output = {
        'Gene': gene,
        'Phenotype': phenotype,
        'Predicted Phenotype_x': phenotype,
        'Activity Score_x': gene_phenotype_row['Activity Score_x'].values[0],
        'EHR Priority Result Notation': gene_phenotype_row['EHR Priority Result Notation'].values[0],
        'Consultation Text': gene_phenotype_row['Consultation Text'].values[0],
        'Allele 1 Function': gene_phenotype_row['Allele 1 Function'].values[0],
        'Allele 2 Function': gene_phenotype_row['Allele 2 Function'].values[0],
        'Activity Value Allele 1': gene_phenotype_row['Activity Value Allele 1'].values[0],
        'Activity Value Allele 2': gene_phenotype_row['Activity Value Allele 2'].values[0],
        'Description': gene_phenotype_row['Description'].values[0]
    }
    return output


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gene = request.form['gene']
    phenotype = request.form['phenotype']
    
    # Call the prediction function
    prediction_details = predict_with_gene_phenotype(gene, phenotype, combined_data, model, scaler, label_encoders, target_encoder)
    
    return render_template('result.html', gene=gene, phenotype=phenotype, result=prediction_details)

if __name__ == '__main__':
    app.run(debug=True)
