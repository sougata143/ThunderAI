import React from 'react';

export const LoadingSkeleton = () => {
  return (
    <div className="loading-skeleton">
      {[...Array(3)].map((_, index) => (
        <div key={index} className="skeleton-card">
          <div className="skeleton-header"></div>
          <div className="skeleton-content">
            <div className="skeleton-line"></div>
            <div className="skeleton-line"></div>
            <div className="skeleton-metrics">
              <div className="skeleton-line"></div>
              <div className="skeleton-line"></div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}; 