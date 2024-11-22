import React from 'react';
import ErrorBoundary from '../common/ErrorBoundary';
import RealTimeMetrics from './RealTimeMetrics';
import { useParams } from 'react-router-dom';

const RealtimeMetricsDashboard = () => {
  const { modelId } = useParams();

  return (
    <ErrorBoundary>
      <RealTimeMetrics modelId={modelId} />
    </ErrorBoundary>
  );
};

export default RealtimeMetricsDashboard; 