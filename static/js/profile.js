document.addEventListener('DOMContentLoaded', function() {
    // Edit Profile Button
    const editProfileBtn = document.getElementById('editProfileBtn');
    const editProfileModal = new bootstrap.Modal(document.getElementById('editProfileModal'));
    const saveProfileBtn = document.getElementById('saveProfileBtn');
    const profilePicture = document.getElementById('profilePicture');
    const editProfilePicture = document.getElementById('editProfilePicture');

    editProfileBtn.addEventListener('click', () => {
        editProfileModal.show();
    });

    // Handle profile picture preview
    editProfilePicture.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                profilePicture.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });

    // Save Profile Changes
    saveProfileBtn.addEventListener('click', async () => {
        const formData = new FormData();
        formData.append('username', document.getElementById('editUsername').value);
        formData.append('email', document.getElementById('editEmail').value);
        formData.append('current_password', document.getElementById('currentPassword').value);
        
        const newPassword = document.getElementById('newPassword').value;
        if (newPassword) {
            formData.append('new_password', newPassword);
        }

        const profilePicture = document.getElementById('editProfilePicture').files[0];
        if (profilePicture) {
            formData.append('profile_picture', profilePicture);
        }

        try {
            const response = await fetch('/api/profile/update', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (response.ok) {
                // Update the profile information on the page
                document.getElementById('username').textContent = data.username;
                document.getElementById('email').textContent = data.email;
                if (data.profile_picture_url) {
                    document.getElementById('profilePicture').src = data.profile_picture_url;
                }
                editProfileModal.hide();
                showToast('Success', 'Profile updated successfully!', 'success');
            } else {
                showToast('Error', data.message || 'Failed to update profile', 'error');
            }
        } catch (error) {
            showToast('Error', 'An error occurred while updating profile', 'error');
        }
    });

    // Delete Project
    document.querySelectorAll('.delete-project').forEach(button => {
        button.addEventListener('click', async (e) => {
            const projectId = e.target.dataset.projectId;
            if (confirm('Are you sure you want to delete this project?')) {
                try {
                    const response = await fetch(`/api/projects/${projectId}`, {
                        method: 'DELETE'
                    });

                    if (response.ok) {
                        // Remove the project from the list
                        e.target.closest('.list-group-item').remove();
                        showToast('Success', 'Project deleted successfully!', 'success');
                    } else {
                        const data = await response.json();
                        showToast('Error', data.message || 'Failed to delete project', 'error');
                    }
                } catch (error) {
                    showToast('Error', 'An error occurred while deleting project', 'error');
                }
            }
        });
    });

    // Helper function to show toast notifications
    function showToast(title, message, type) {
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type === 'success' ? 'success' : 'danger'} border-0`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <strong>${title}:</strong> ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        document.body.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }
});
