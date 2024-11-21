import React, { useState, useEffect, useRef } from 'react';
import {
  Paper,
  Box,
  TextField,
  IconButton,
  Typography,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Drawer,
  Badge
} from '@mui/material';
import {
  Send as SendIcon,
  Chat as ChatIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import { formatDistanceToNow } from 'date-fns';
import CollaborationService from '../../services/collaborationService';

function Chat() {
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    CollaborationService.subscribe('chat_message', handleNewMessage);
    return () => CollaborationService.unsubscribe('chat_message', handleNewMessage);
  }, []);

  const handleNewMessage = (message) => {
    setMessages(prev => [...prev, message]);
    if (!open) {
      setUnreadCount(prev => prev + 1);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSend = () => {
    if (message.trim()) {
      CollaborationService.sendChatMessage(message);
      setMessage('');
    }
  };

  const toggleDrawer = (isOpen) => {
    setOpen(isOpen);
    if (isOpen) {
      setUnreadCount(0);
    }
  };

  return (
    <>
      <IconButton
        onClick={() => toggleDrawer(true)}
        sx={{ position: 'fixed', bottom: 16, right: 16 }}
      >
        <Badge badgeContent={unreadCount} color="error">
          <ChatIcon />
        </Badge>
      </IconButton>

      <Drawer
        anchor="right"
        open={open}
        onClose={() => toggleDrawer(false)}
      >
        <Paper sx={{ width: 320, height: '100%', display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center' }}>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>Chat</Typography>
            <IconButton onClick={() => toggleDrawer(false)}>
              <CloseIcon />
            </IconButton>
          </Box>

          <List sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
            {messages.map((msg, index) => (
              <ListItem key={index} alignItems="flex-start">
                <ListItemAvatar>
                  <Avatar alt={msg.user.name} src={msg.user.avatar}>
                    {msg.user.name.charAt(0)}
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={msg.user.name}
                  secondary={
                    <>
                      <Typography component="span" variant="body2">
                        {msg.content}
                      </Typography>
                      <Typography component="span" variant="caption" sx={{ display: 'block' }}>
                        {formatDistanceToNow(new Date(msg.timestamp), { addSuffix: true })}
                      </Typography>
                    </>
                  }
                />
              </ListItem>
            ))}
            <div ref={messagesEndRef} />
          </List>

          <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Type a message..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
              InputProps={{
                endAdornment: (
                  <IconButton onClick={handleSend}>
                    <SendIcon />
                  </IconButton>
                )
              }}
            />
          </Box>
        </Paper>
      </Drawer>
    </>
  );
}

export default Chat; 