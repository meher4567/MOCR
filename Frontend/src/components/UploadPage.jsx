import PropTypes from 'prop-types'; // For prop type-checking
import { useState } from 'react'; // For handling file upload state
import axios from 'axios'; // For making API requests
import './UploadPage.css'; // Importing the CSS file for styling

const UploadPage = () => {
  const [uploadMode, setUploadMode] = useState('single'); // Track the selected upload mode
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [errorMessage, setErrorMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [downloadLink, setDownloadLink] = useState('');

  // Handles file selection
  const onFileChange = (event) => {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
      setSelectedFiles(files);
      setErrorMessage('');
      setDownloadLink('');
    }
  };

  // Handles form submission
  const onFormSubmit = async (event) => {
    event.preventDefault();
    if (selectedFiles.length === 0) {
      setErrorMessage('Please select a file or a folder before submitting.');
      return;
    }

    try {
      setLoading(true);
      setErrorMessage('');

      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append('images', file); // Append selected files
      });

      // API call to the backend
      const response = await axios.post('http://127.0.0.1:8000/ocr/upload/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob', // Expecting a zip file as the response
      });

      // Create a URL for downloading the zip file
      const blob = new Blob([response.data], { type: 'application/zip' });
      const url = window.URL.createObjectURL(blob);
      setDownloadLink(url); // Store the download URL
    } catch (error) {
      setErrorMessage('An error occurred while uploading files. Please try again.');
      console.error('Upload error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-page">
      <h2>Upload Manga Image(s)</h2>
      {/* Radio buttons to toggle between single image and folder upload */}
      <div className="upload-mode-selector">
        <label>
          <input
            type="radio"
            name="uploadMode"
            value="single"
            checked={uploadMode === 'single'}
            onChange={() => setUploadMode('single')}
          />
          Single Image
        </label>
        <label>
          <input
            type="radio"
            name="uploadMode"
            value="folder"
            checked={uploadMode === 'folder'}
            onChange={() => setUploadMode('folder')}
          />
          Folder of Images
        </label>
      </div>
      <form className="upload-form" onSubmit={onFormSubmit}>
        {/* File input */}
        <input
          type="file"
          accept="image/*"
          multiple={uploadMode === 'folder'}
          onChange={onFileChange}
          className="file-input"
          webkitdirectory={uploadMode === 'folder' ? '' : undefined} // Allow folder selection only in folder mode
        />
        {errorMessage && <p className="error-message">{errorMessage}</p>}
        {/* Display selected files */}
        <ul className="file-list">
          {selectedFiles.map((file, index) => (
            <li key={index}>{file.name}</li>
          ))}
        </ul>
        {/* Submit button */}
        <button type="submit" className="submit-button" disabled={loading}>
          {loading ? 'Uploading...' : 'Submit'}
        </button>
      </form>
      {/* Download link for the zip file */}
      {downloadLink && (
        <div className="download-section">
          <a href={downloadLink} download="translated_images.zip">
            Download Translated Images
          </a>
        </div>
      )}
    </div>
  );
};

// Prop type validation
UploadPage.propTypes = {
  apiUrl: PropTypes.string,
};

export default UploadPage;
