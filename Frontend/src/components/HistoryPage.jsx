import PropTypes from 'prop-types'; // For prop type-checking
import './HistoryPage.css'; // Importing the CSS file for styling

const HistoryPage = ({ historyData }) => {
  if (!historyData || historyData.length === 0) {
    return (
      <div className="history-page">
        <p>No history available. Start uploading images to view history.</p>
      </div>
    );
  }

  return (
    <div className="history-page">
      <h2>Translation History</h2>
      <ul className="history-list">
        {historyData.map((item, index) => (
          <li key={index} className="history-item">
            <p>
              <strong>Image:</strong> {item.imageName}
            </p>
            <p>
              <strong>Status:</strong> {item.status}
            </p>
            <button
              onClick={() =>
                alert(`Viewing details for ${item.imageName}`) // Placeholder logic
              }
              className="view-details-button"
            >
              View Details
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
};

// Prop type validation
HistoryPage.propTypes = {
  historyData: PropTypes.arrayOf(
    PropTypes.shape({
      imageName: PropTypes.string.isRequired,
      status: PropTypes.string.isRequired,
    })
  ),
};

export default HistoryPage;
