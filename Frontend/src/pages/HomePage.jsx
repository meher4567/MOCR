import PropTypes from 'prop-types'; // For prop type-checking
import './HomePage.css'; // Importing the CSS file for styling

const HomePage = ({ navigateToUpload }) => {
  return (
    <div className="home-page">
      {/* Welcome message and app introduction */}
      <div className="welcome-section">
        <h2>Welcome to Manga OCR & Translation</h2>
        <p>
          This app allows you to upload manga images, extract text using OCR, 
          and translate it into your preferred language.
        </p>
      </div>

      {/* Call-to-action button */}
      <div className="cta-section">
        <button onClick={navigateToUpload} className="cta-button">
          Get Started
        </button>
      </div>
    </div>
  );
};

// Prop type validation
HomePage.propTypes = {
  navigateToUpload: PropTypes.func.isRequired, // Ensures the function is provided and is callable
};

export default HomePage;
