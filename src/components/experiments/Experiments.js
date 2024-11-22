import React, { useState, useEffect } from 'react';
import { LoadingSkeleton } from '../common/LoadingSkeleton';
import '../../styles/LoadingSkeleton.css';
import '../../styles/Experiments.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

const Experiments = () => {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchExperiments = async () => {
      let response;
      try {
        setLoading(true);
        response = await fetch(`${BACKEND_URL}/api/v1/experiments`, {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          },
          credentials: 'include'
        });

        if (response.status === 404) {
          throw new Error('Experiments endpoint not found. Please check the API configuration.');
        }

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.message || `Server error: ${response.status}`);
        }

        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
          throw new Error('Invalid response format from server');
        }

        const data = await response.json();
        
        // Handle different response formats
        const experimentsList = Array.isArray(data) ? data : 
                              data.experiments ? data.experiments :
                              data.data ? data.data : [];
        
        if (!Array.isArray(experimentsList)) {
          throw new Error('Invalid data format received from server');
        }

        setExperiments(experimentsList);
      } catch (error) {
        console.error('Error fetching experiments:', error);
        let errorMessage = error.message;
        
        // Provide more user-friendly error messages
        if (error.message.includes('Failed to fetch') || error.message.includes('Load failed')) {
          errorMessage = `Unable to connect to the server at ${BACKEND_URL}. Please check if:
          1. The backend server is running
          2. CORS is properly configured
          3. The server is accessible at ${BACKEND_URL}`;
        } else if (response?.status === 401) {
          errorMessage = 'Please log in to view experiments.';
        } else if (response?.status === 403) {
          errorMessage = 'You do not have permission to view experiments.';
        }
        
        setError(errorMessage);
      } finally {
        setLoading(false);
      }
    };

    fetchExperiments();
  }, []);

  if (loading) {
    return <LoadingSkeleton />;
  }

  if (error) {
    return (
      <div className="error-container">
        <h3>Error loading experiments</h3>
        <p style={{ whiteSpace: 'pre-line' }}>{error}</p>
        <div className="error-actions">
          <button 
            onClick={() => window.location.reload()} 
            className="retry-button"
          >
            Retry
          </button>
          <button 
            onClick={() => setError(null)} 
            className="dismiss-button"
          >
            Dismiss
          </button>
          <button
            onClick={() => window.location.href = '/'}
            className="home-button"
          >
            Go to Home
          </button>
        </div>
      </div>
    );
  }

  if (!experiments || experiments.length === 0) {
    return (
      <div className="no-experiments">
        <h3>No experiments found</h3>
        <p>Start a new experiment to see it here.</p>
        <button 
          onClick={() => window.location.href = '/experiments/new'} 
          className="create-experiment-button"
        >
          Create New Experiment
        </button>
      </div>
    );
  }

  return (
    <div className="experiments-container">
      {experiments.map((experiment) => (
        <div key={experiment.id || experiment._id} className="experiment-card">
          <h3>{experiment.name || experiment.title || 'Unnamed Experiment'}</h3>
          <div className="experiment-details">
            <p>Status: <span className={`status-${experiment.status?.toLowerCase()}`}>
              {experiment.status || 'Unknown'}
            </span></p>
            <p>Created: {new Date(experiment.created_at || experiment.createdAt).toLocaleDateString()}</p>
            {experiment.metrics && (
              <div className="metrics">
                <p>Loss: {typeof experiment.metrics.loss === 'number' ? 
                  experiment.metrics.loss.toFixed(4) : 'N/A'}</p>
                <p>Accuracy: {typeof experiment.metrics.accuracy === 'number' ? 
                  experiment.metrics.accuracy.toFixed(4) : 'N/A'}</p>
              </div>
            )}
            {experiment.description && (
              <p className="experiment-description">
                {experiment.description}
              </p>
            )}
          </div>
          <div className="experiment-actions">
            <button 
              onClick={() => window.location.href = `/experiments/${experiment.id || experiment._id}`}
              className="view-details-button"
            >
              View Details
            </button>
            {experiment.status === 'running' && (
              <button
                onClick={() => window.location.href = `/experiments/${experiment.id || experiment._id}/monitor`}
                className="monitor-button"
              >
                Monitor Progress
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default Experiments; 