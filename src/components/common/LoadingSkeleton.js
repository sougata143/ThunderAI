import React from 'react';
import { Skeleton, Paper, Grid } from '@mui/material';

function LoadingSkeleton({ type = 'chart' }) {
  const renderChartSkeleton = () => (
    <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
      <Skeleton variant="text" width="30%" height={40} />
      <Skeleton variant="rectangular" width="100%" height={300} />
    </Paper>
  );

  const renderTableSkeleton = () => (
    <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
      <Skeleton variant="text" width="30%" height={40} />
      <Grid container spacing={1}>
        {[...Array(5)].map((_, i) => (
          <Grid item xs={12} key={i}>
            <Skeleton variant="rectangular" height={40} />
          </Grid>
        ))}
      </Grid>
    </Paper>
  );

  return (
    <div>
      {type === 'chart' && renderChartSkeleton()}
      {type === 'table' && renderTableSkeleton()}
      {type === 'combined' && (
        <>
          {renderChartSkeleton()}
          {renderTableSkeleton()}
        </>
      )}
    </div>
  );
}

export default LoadingSkeleton; 