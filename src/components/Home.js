import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';

function Home() {
  const [models, setModels] = useState([]);

  useEffect(() => {
    axios.get('/api/v1/models')
      .then(response => setModels(response.data))
      .catch(error => console.error('Error fetching models:', error));
  }, []);

  return (
    <div>
      <h1>ThunderAI Models</h1>
      <ul>
        {models.map(model => (
          <li key={model.name}>
            <Link to={`/model/${model.name}`}>{model.name}</Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Home; 