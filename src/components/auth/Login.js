import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { login, loginAsGuest } from '../../store/slices/authSlice';
import { 
    Container, 
    Paper, 
    Typography, 
    Box, 
    TextField, 
    Button, 
    Divider,
    Alert
} from '@mui/material';

const Login = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate();
    const dispatch = useDispatch();

    const { token, user } = useSelector((state) => state.auth);

    useEffect(() => {
        if (token && user) {
            navigate('/dashboard');
        }
    }, [token, user, navigate]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        
        try {
            const result = await dispatch(login({ email, password })).unwrap();
            console.log('Login response:', result);
            
            if (result.access_token) {
                navigate('/dashboard');
            } else {
                setError('Invalid response from server');
            }
        } catch (err) {
            console.error('Login error:', err);
            setError(err.message || 'Invalid credentials. Please try again.');
        }
    };

    const handleGuestLogin = async () => {
        const result = await dispatch(loginAsGuest());
        if (!result.error) {
            navigate('/dashboard');
        }
    };

    return (
        <Container component="main" maxWidth="xs">
            <Box
                sx={{
                    marginTop: 8,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                }}
            >
                <Paper elevation={3} sx={{ p: 4, width: '100%' }}>
                    <Typography component="h1" variant="h5" align="center" gutterBottom>
                        Welcome Back
                    </Typography>
                    <Typography variant="body2" color="textSecondary" align="center" sx={{ mb: 3 }}>
                        Sign in to continue to ThunderAI
                    </Typography>

                    {error && (
                        <Alert severity="error" sx={{ mb: 3 }}>
                            {error}
                        </Alert>
                    )}

                    <Box component="form" onSubmit={handleSubmit} noValidate>
                        <TextField
                            margin="normal"
                            required
                            fullWidth
                            id="email"
                            label="Email Address"
                            name="email"
                            autoComplete="email"
                            autoFocus
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                        />
                        <TextField
                            margin="normal"
                            required
                            fullWidth
                            name="password"
                            label="Password"
                            type="password"
                            id="password"
                            autoComplete="current-password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                        />
                        <Button
                            type="submit"
                            fullWidth
                            variant="contained"
                            sx={{ mt: 3, mb: 2 }}
                        >
                            Sign In
                        </Button>

                        <Divider sx={{ my: 2 }}>or</Divider>

                        <Button
                            fullWidth
                            variant="outlined"
                            onClick={handleGuestLogin}
                            sx={{ mb: 2 }}
                        >
                            Continue as Guest
                        </Button>

                        <Box sx={{ mt: 2, textAlign: 'center' }}>
                            <Typography variant="body2" color="textSecondary">
                                Don't have an account?{' '}
                                <Link 
                                    to="/register"
                                    style={{ 
                                        color: 'primary.main', 
                                        textDecoration: 'none' 
                                    }}
                                >
                                    Sign up
                                </Link>
                            </Typography>
                        </Box>
                    </Box>
                </Paper>
            </Box>
        </Container>
    );
};

export default Login;