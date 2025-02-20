
import PropTypes from 'prop-types'; // For prop type-checking
import './OCRResultsPage.css'; // Importing the CSS file for styling

const OCRResultsPage = ({ ocrResults, goToUpload }) => {
  if (!ocrResults) {
    return (
      <div className="ocr-results-page">
        <p>No OCR results available. Please upload an image first.</p>
        <button onClick={goToUpload} className="back-to-upload-button">
          Upload Image
        </button>
      </div>
    );
  }

  return (
    <div className="ocr-results-page">
      <h2>OCR and Translation Results</h2>
      <div className="results-section">
        <div className="result">
          <h3>Extracted Text</h3>
          <p>{ocrResults.extractedText}</p>
        </div>
        <div className="result">
          <h3>Translated Text</h3>
          <p>{ocrResults.translatedText}</p>
        </div>
      </div>
      <div className="actions">
        <button onClick={goToUpload} className="upload-again-button">
          Upload Another Image
        </button>
        <button
          onClick={() =>
            alert('Download feature coming soon!') // Placeholder for download logic
          }
          className="download-button"
        >
          Download Results
        </button>
      </div>
    </div>
  );
};

// Prop type validation
OCRResultsPage.propTypes = {
  ocrResults: PropTypes.shape({
    extractedText: PropTypes.string.isRequired,
    translatedText: PropTypes.string.isRequired,
  }),
  goToUpload: PropTypes.func.isRequired,
};

export default OCRResultsPage;
