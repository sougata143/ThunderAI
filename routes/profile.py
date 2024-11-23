from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
import os
from models.user import User
from models.project import Project
from database import db

profile_bp = Blueprint('profile', __name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@profile_bp.route('/profile')
@login_required
def profile():
    # Get user's projects
    projects = Project.query.filter_by(user_id=current_user.id).all()
    return render_template('profile.html', projects=projects)

@profile_bp.route('/api/profile/update', methods=['POST'])
@login_required
def update_profile():
    try:
        # Get form data
        username = request.form.get('username')
        email = request.form.get('email')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')

        # Verify current password
        if not check_password_hash(current_user.password_hash, current_password):
            return jsonify({'message': 'Current password is incorrect'}), 400

        # Check if username is taken
        if username != current_user.username:
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                return jsonify({'message': 'Username is already taken'}), 400

        # Check if email is taken
        if email != current_user.email:
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                return jsonify({'message': 'Email is already taken'}), 400

        # Update user information
        current_user.username = username
        current_user.email = email

        # Update password if provided
        if new_password:
            current_user.password_hash = generate_password_hash(new_password)

        # Handle profile picture upload
        profile_picture_url = None
        if 'profile_picture' in request.files:
            file = request.files['profile_picture']
            if file and allowed_file(file.filename):
                filename = secure_filename(f"{current_user.id}_{file.filename}")
                upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'profile_pictures')
                os.makedirs(upload_path, exist_ok=True)
                file_path = os.path.join(upload_path, filename)
                file.save(file_path)
                current_user.profile_picture = filename
                profile_picture_url = f"/static/uploads/profile_pictures/{filename}"

        db.session.commit()

        return jsonify({
            'message': 'Profile updated successfully',
            'username': current_user.username,
            'email': current_user.email,
            'profile_picture_url': profile_picture_url
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 500

@profile_bp.route('/api/projects/<int:project_id>', methods=['DELETE'])
@login_required
def delete_project(project_id):
    try:
        project = Project.query.get(project_id)
        if not project:
            return jsonify({'message': 'Project not found'}), 404
        
        if project.user_id != current_user.id:
            return jsonify({'message': 'Unauthorized'}), 403

        db.session.delete(project)
        db.session.commit()

        return jsonify({'message': 'Project deleted successfully'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 500
