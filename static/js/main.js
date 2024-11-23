// Enable Bootstrap tooltips
document.addEventListener('DOMContentLoaded', function() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    })

    // Auto-hide alerts after 5 seconds
    var alerts = document.querySelectorAll('.alert:not(.alert-permanent)')
    alerts.forEach(function(alert) {
        setTimeout(function() {
            var bsAlert = new bootstrap.Alert(alert)
            bsAlert.close()
        }, 5000)
    })

    // Add fade-in class to main content
    var main = document.querySelector('main')
    if (main) {
        main.classList.add('fade-in')
    }
})

// Handle form submissions with fetch API
document.addEventListener('submit', function(e) {
    var form = e.target
    if (form.hasAttribute('data-fetch')) {
        e.preventDefault()
        
        var submitBtn = form.querySelector('[type="submit"]')
        if (submitBtn) {
            submitBtn.disabled = true
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...'
        }

        fetch(form.action, {
            method: form.method,
            body: new FormData(form),
            credentials: 'same-origin'
        })
        .then(response => response.json())
        .then(data => {
            if (data.redirect) {
                window.location.href = data.redirect
            } else if (data.message) {
                showAlert(data.message, data.category || 'info')
            }
        })
        .catch(error => {
            showAlert('An error occurred. Please try again.', 'danger')
            console.error('Error:', error)
        })
        .finally(() => {
            if (submitBtn) {
                submitBtn.disabled = false
                submitBtn.innerHTML = submitBtn.getAttribute('data-original-text') || 'Submit'
            }
        })
    }
})

// Show alert function
function showAlert(message, category = 'info') {
    const alertHtml = `
        <div class="alert alert-${category} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `
    const alertContainer = document.querySelector('.container.mt-3')
    if (alertContainer) {
        alertContainer.insertAdjacentHTML('beforeend', alertHtml)
        
        // Auto-hide after 5 seconds
        const newAlert = alertContainer.lastElementChild
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(newAlert)
            bsAlert.close()
        }, 5000)
    }
}
