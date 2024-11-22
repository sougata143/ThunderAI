import React, { useEffect, useState, useRef, useCallback } from 'react';
import WebSocketManager from '../../utils/WebSocketManager';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const RealTimeMetrics = ({ modelId }) => {
  const [metrics, setMetrics] = useState({
    loss: [],
    accuracy: [],
    epochs: [],
    timestamp: []
  });
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [errorDetails, setErrorDetails] = useState(null);
  const wsManager = useRef(null);
  const chartRef = useRef(null);

  const updateMetrics = useCallback((newData) => {
    setMetrics(prevMetrics => {
      // Ensure we don't add duplicate epochs
      if (prevMetrics.epochs.includes(newData.epoch)) {
        return prevMetrics;
      }

      return {
        loss: [...prevMetrics.loss, newData.loss],
        accuracy: [...prevMetrics.accuracy, newData.accuracy],
        epochs: [...prevMetrics.epochs, newData.epoch],
        timestamp: [...prevMetrics.timestamp, new Date().getTime()]
      };
    });
  }, []);

  useEffect(() => {
    if (!modelId) {
      console.error('No modelId provided');
      setErrorDetails('No model ID provided');
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const WS_URL = `${protocol}//${window.location.host}/api/v1/models/ws/training/${modelId}`;
    
    console.log('Initializing WebSocket connection to:', WS_URL);

    wsManager.current = new WebSocketManager(WS_URL, {
      reconnectAttempts: 10,
      reconnectInterval: 2000,
      onMessage: (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Received training data:', data);
          
          if (data.type === 'training_metrics' && data.loss !== undefined && data.accuracy !== undefined) {
            updateMetrics({
              loss: parseFloat(data.loss),
              accuracy: parseFloat(data.accuracy),
              epoch: data.epoch || metrics.epochs.length + 1
            });
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
          setErrorDetails(`Failed to parse message: ${error.message}`);
        }
      },
      onError: (error) => {
        setConnectionStatus('error');
        setErrorDetails(error.message || 'Unknown WebSocket error');
        console.error('WebSocket error:', error);
      },
      onClose: (event) => {
        console.log('WebSocket closed with code:', event.code);
        if (event.code === 1000) {
          setConnectionStatus('completed');
        } else {
          setConnectionStatus('disconnected');
          setErrorDetails(`Connection closed (Code: ${event.code}${event.reason ? `, Reason: ${event.reason}` : ''})`);
        }
      },
      onOpen: () => {
        setConnectionStatus('connected');
        setErrorDetails(null);
        console.log('WebSocket connected successfully');
        
        // Request initial state if needed
        wsManager.current.send(JSON.stringify({
          type: 'request_current_state',
          modelId: modelId
        }));
      }
    });

    return () => {
      console.log('Cleaning up WebSocket connection');
      if (wsManager.current) {
        wsManager.current.close();
      }
    };
  }, [modelId, updateMetrics]);

  const chartData = {
    labels: metrics.epochs,
    datasets: [
      {
        label: 'Loss',
        data: metrics.loss,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        fill: true,
        tension: 0.1,
        pointRadius: 2
      },
      {
        label: 'Accuracy',
        data: metrics.accuracy,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.1)',
        fill: true,
        tension: 0.1,
        pointRadius: 2
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1.0,
        ticks: {
          callback: value => value.toFixed(2)
        }
      },
      x: {
        title: {
          display: true,
          text: 'Epoch'
        }
      }
    },
    plugins: {
      title: {
        display: true,
        text: 'Training Progress'
      },
      legend: {
        position: 'top'
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += context.parsed.y.toFixed(4);
            }
            return label;
          }
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  return (
    <div className="real-time-metrics">
      <div className={`connection-status ${connectionStatus}`}>
        Status: {connectionStatus}
        {errorDetails && (
          <div className="error-details">
            Error: {errorDetails}
          </div>
        )}
      </div>
      
      {connectionStatus === 'error' && (
        <div className="error-message">
          Connection error. Attempting to reconnect...
        </div>
      )}
      
      {connectionStatus === 'connecting' && (
        <div className="connecting-message">
          Connecting to training service...
        </div>
      )}

      {(connectionStatus === 'connected' || connectionStatus === 'completed') && (
        <div className="metrics-display">
          <div className="chart-container" style={{ height: '400px', width: '100%', padding: '20px' }}>
            <Line ref={chartRef} data={chartData} options={chartOptions} />
          </div>
          
          <div className="metrics-summary">
            <div className="metric">
              Latest Loss: {metrics.loss[metrics.loss.length - 1]?.toFixed(4) || 'N/A'}
            </div>
            <div className="metric">
              Latest Accuracy: {metrics.accuracy[metrics.accuracy.length - 1]?.toFixed(4) || 'N/A'}
            </div>
            <div className="metric">
              Current Epoch: {metrics.epochs[metrics.epochs.length - 1] || 'N/A'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RealTimeMetrics; 