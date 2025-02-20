import { useState } from 'react'; // For managing the current page state
import Navbar from './components/Navbar'; // Navigation bar
import HomePage from './pages/HomePage'; // Home page
import UploadPage from './components/UploadPage'; // Upload page
import OCRResultsPage from './components/OCRResultsPage'; // OCR Results page
import HistoryPage from './components/HistoryPage'; // History page
import './App.css';

// Main App Component
function App() {
  // State to track the current page
  const [currentPage, setCurrentPage] = useState('home');

  // State to manage the uploaded file, OCR results, and history data
  const [uploadedFile, setUploadedFile] = useState(null);
  const [ocrResults, setOcrResults] = useState(null);
  const [historyData, setHistoryData] = useState([]);

  // Function to handle file upload from the UploadPage
  const handleFileUpload = (file) => {
    setUploadedFile(file); // Save the uploaded file
    setCurrentPage('results'); // Navigate to the results page

    // Simulate OCR processing (replace with an actual API call in a real app)
    const ocrOutput = {
      extractedText: 'Sample OCR text from image.',
      translatedText: 'Sample translated text.',
    };
    setOcrResults(ocrOutput); // Save OCR results

    // Add the uploaded file details to history
    setHistoryData((prevHistory) => [
      ...prevHistory,
      { imageName: file.name, status: 'Completed' },
    ]);
  };

  // Function to navigate to the upload page
  const goToUpload = () => {
    setCurrentPage('upload');
  };

  // Function to dynamically render the appropriate page
  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage navigateToUpload={() => setCurrentPage('upload')} />;
      case 'upload':
        return <UploadPage handleFileUpload={handleFileUpload} />;
      case 'results':
        return (
          <OCRResultsPage
            ocrResults={ocrResults}
            goToUpload={goToUpload}
          />
        );
      case 'history':
        return <HistoryPage historyData={historyData} />;
      default:
        return <HomePage navigateToUpload={() => setCurrentPage('upload')} />;
    }
  };

  return (
    <div className="app-container">
      {/* Navbar for page navigation */}
      <Navbar setCurrentPage={setCurrentPage} />

      {/* Main content area dynamically renders pages */}
      <main className="app-content">{renderPage()}</main>
    </div>
  );
}

export default App;
