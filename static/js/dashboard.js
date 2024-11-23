// Load projects when the page loads
document.addEventListener('DOMContentLoaded', () => {
    loadProjects();
    loadRecentActivity();
    setupProjectCreation();
    loadMetrics();
});

// Load user's projects
async function loadProjects() {
    try {
        const response = await apiRequest('/projects/me');
        const projects = response.projects;
        
        const projectsList = document.getElementById('projectsList');
        projectsList.innerHTML = ''; // Clear existing projects
        
        if (projects.length === 0) {
            projectsList.innerHTML = `
                <div class="col-12 text-center py-5">
                    <p class="text-muted">No projects yet. Create your first project!</p>
                </div>
            `;
            return;
        }
        
        projects.forEach(project => {
            const projectCard = createProjectCard(project);
            projectsList.appendChild(projectCard);
        });
    } catch (error) {
        showAlert('Failed to load projects', 'danger');
    }
}

// Create a project card element
function createProjectCard(project) {
    const col = document.createElement('div');
    col.className = 'col-md-4 mb-4';
    
    col.innerHTML = `
        <div class="card h-100 project-card">
            <div class="card-body">
                <h5 class="card-title">${project.name}</h5>
                <p class="card-text">${project.description || 'No description'}</p>
                <div class="text-muted small mb-2">
                    Created: ${new Date(project.created_at).toLocaleDateString()}
                </div>
            </div>
            <div class="card-footer bg-transparent">
                <div class="d-flex justify-content-between align-items-center">
                    <button class="btn btn-sm btn-primary" onclick="openProject('${project.id}')">
                        Open Project
                    </button>
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteProject('${project.id}')">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
        </div>
    `;
    
    return col;
}

// Load recent activity
async function loadRecentActivity() {
    try {
        const response = await apiRequest('/activity/recent');
        const activities = response.activities;
        
        const activityList = document.getElementById('activityList');
        activityList.innerHTML = ''; // Clear existing activities
        
        if (activities.length === 0) {
            activityList.innerHTML = `
                <p class="text-muted text-center py-3">No recent activity</p>
            `;
            return;
        }
        
        activities.forEach(activity => {
            const activityItem = document.createElement('div');
            activityItem.className = 'border-bottom py-2';
            activityItem.innerHTML = `
                <div class="d-flex align-items-center">
                    <div class="flex-grow-1">
                        <p class="mb-1">${activity.description}</p>
                        <small class="text-muted">${new Date(activity.timestamp).toLocaleString()}</small>
                    </div>
                </div>
            `;
            activityList.appendChild(activityItem);
        });
    } catch (error) {
        showAlert('Failed to load recent activity', 'danger');
    }
}

// Setup project creation
function setupProjectCreation() {
    const createProjectBtn = document.getElementById('createProjectBtn');
    const projectForm = document.getElementById('newProjectForm');
    
    createProjectBtn.addEventListener('click', async () => {
        if (!validateForm(projectForm)) return;
        
        const projectData = {
            name: document.getElementById('projectName').value,
            description: document.getElementById('projectDescription').value
        };
        
        try {
            setLoading(createProjectBtn, true);
            await apiRequest('/projects', {
                method: 'POST',
                body: JSON.stringify(projectData)
            });
            
            // Close modal and reset form
            const modal = bootstrap.Modal.getInstance(document.getElementById('newProjectModal'));
            modal.hide();
            projectForm.reset();
            
            // Reload projects
            loadProjects();
            showAlert('Project created successfully', 'success');
        } catch (error) {
            showAlert('Failed to create project', 'danger');
        } finally {
            setLoading(createProjectBtn, false);
        }
    });
}

// Open a project
function openProject(projectId) {
    window.location.href = `/project/${projectId}`;
}

// Delete a project
async function deleteProject(projectId) {
    if (!confirm('Are you sure you want to delete this project?')) return;
    
    try {
        await apiRequest(`/projects/${projectId}`, {
            method: 'DELETE'
        });
        
        loadProjects();
        showAlert('Project deleted successfully', 'success');
    } catch (error) {
        showAlert('Failed to delete project', 'danger');
    }
}

// Chart instances
let accuracyChart = null;
let lossChart = null;
let metricsTimelineChart = null;

// Chart colors
const colors = {
    primary: '#4CAF50',
    secondary: '#2196F3',
    warning: '#FFC107',
    danger: '#F44336',
    success: '#4CAF50',
    info: '#00BCD4'
};

// Chart options
const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            position: 'top',
        }
    },
    scales: {
        x: {
            grid: {
                display: false
            }
        },
        y: {
            beginAtZero: true,
            grid: {
                borderDash: [2],
                borderDashOffset: [2],
                drawBorder: false,
                zeroLineColor: "rgba(0,0,0,0)",
                drawTicks: false
            },
            ticks: {
                padding: 10
            }
        }
    }
};

// Load metrics data
async function loadMetrics() {
    try {
        const response = await fetch('/api/v1/metrics/overview', {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to fetch metrics');
        }
        
        const data = await response.json();
        updateDashboard(data);
    } catch (error) {
        console.error('Error loading metrics:', error);
        // Show error message to user
        showError('Failed to load metrics data. Please try again later.');
    }
}

// Update dashboard with metrics data
function updateDashboard(data) {
    // Update summary cards
    document.getElementById('avgTrainAcc').textContent = (data.summary.avg_train_accuracy * 100).toFixed(2) + '%';
    document.getElementById('avgValAcc').textContent = (data.summary.avg_val_accuracy * 100).toFixed(2) + '%';
    document.getElementById('avgF1').textContent = (data.summary.avg_f1_score * 100).toFixed(2) + '%';
    document.getElementById('totalRecords').textContent = data.summary.total_records;
    
    // Process data for charts
    const timestamps = data.metrics.map(m => new Date(m.timestamp).toLocaleDateString());
    const trainAcc = data.metrics.map(m => m.train_accuracy);
    const valAcc = data.metrics.map(m => m.val_accuracy);
    const trainLoss = data.metrics.map(m => m.train_loss);
    const valLoss = data.metrics.map(m => m.val_loss);
    const f1Scores = data.metrics.map(m => m.f1_score);
    const precision = data.metrics.map(m => m.precision);
    const recall = data.metrics.map(m => m.recall);
    
    // Update accuracy chart
    updateAccuracyChart(timestamps, trainAcc, valAcc);
    
    // Update loss chart
    updateLossChart(timestamps, trainLoss, valLoss);
    
    // Update metrics timeline
    updateMetricsTimeline(timestamps, f1Scores, precision, recall);
}

// Update accuracy chart
function updateAccuracyChart(labels, trainAcc, valAcc) {
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    
    if (accuracyChart) {
        accuracyChart.destroy();
    }
    
    accuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Training Accuracy',
                    data: trainAcc,
                    borderColor: colors.primary,
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Validation Accuracy',
                    data: valAcc,
                    borderColor: colors.secondary,
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: commonOptions
    });
}

// Update loss chart
function updateLossChart(labels, trainLoss, valLoss) {
    const ctx = document.getElementById('lossChart').getContext('2d');
    
    if (lossChart) {
        lossChart.destroy();
    }
    
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Training Loss',
                    data: trainLoss,
                    borderColor: colors.danger,
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Validation Loss',
                    data: valLoss,
                    borderColor: colors.warning,
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: commonOptions
    });
}

// Update metrics timeline
function updateMetricsTimeline(labels, f1Scores, precision, recall) {
    const ctx = document.getElementById('metricsTimeline').getContext('2d');
    
    if (metricsTimelineChart) {
        metricsTimelineChart.destroy();
    }
    
    metricsTimelineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'F1 Score',
                    data: f1Scores,
                    borderColor: colors.success,
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Precision',
                    data: precision,
                    borderColor: colors.info,
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Recall',
                    data: recall,
                    borderColor: colors.warning,
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: commonOptions
    });
}

// Show error message
function showError(message) {
    // You can implement your own error display logic here
    console.error(message);
}

// Load metrics when the page loads
document.addEventListener('DOMContentLoaded', () => {
    loadMetrics();
    
    // Refresh metrics every 5 minutes
    setInterval(loadMetrics, 5 * 60 * 1000);
});
