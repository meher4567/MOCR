import PropTypes from 'prop-types'; // For type-checking props
import './Navbar.css'; // Importing external CSS for styling

const Navbar = ({ setCurrentPage }) => {
  // Handles navigation by updating the current page state
  const handleNavigation = (page) => {
    setCurrentPage(page);
  };

  return (
    <nav className="navbar">
      {/* App Title */}
      <div className="navbar-title">
        <h1>Manga OCR</h1>
      </div>

      {/* Navigation Links */}
      <ul className="navbar-links">
        <li onClick={() => handleNavigation('home')} className="navbar-link">
          Home
        </li>
        <li onClick={() => handleNavigation('upload')} className="navbar-link">
          Upload
        </li>
        <li onClick={() => handleNavigation('history')} className="navbar-link">
          History
        </li>
      </ul>
    </nav>
  );
};

// Type-checking for props
Navbar.propTypes = {
  setCurrentPage: PropTypes.func.isRequired,
};

export default Navbar;
