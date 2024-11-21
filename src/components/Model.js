import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';

function Model() {
  const { modelName } = useParams();
  const [modelDetails, setModelDetails] = useState(null);

  useEffect(() => {
    axios.get(`/api/v1/models/${modelName}`)
      .then(response => setModelDetails(response.data))
      .catch(error => console.error('Error fetching model details:', error));
  }, [modelName]);

  if (!modelDetails) return <div>Loading...</div>;

  return (
    <div>
      <h1>{modelDetails.name}</h1>
      <p>Version: {modelDetails.version}</p>
      <p>Accuracy: {modelDetails.metrics.accuracy}</p>
      {/* Add more model details and visualizations here */}
    </div>
  );
}

export default Model; 