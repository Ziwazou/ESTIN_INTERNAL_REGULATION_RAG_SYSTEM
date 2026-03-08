/**
 * ESTIN RAG Chatbot - Frontend Application
 * 
 * Handles chat interactions with the FastAPI backend
 */

// =============================================================================
// Configuration
// =============================================================================

const CONFIG = {
    API_BASE_URL: 'http://localhost:8000',
    ENDPOINTS: {
        ASK: '/api/v1/ask',
        HEALTH: '/health'
    }
};

// =============================================================================
// State Management
// =============================================================================

const state = {
    threadId: null,
    isLoading: false,
    messages: []
};

// =============================================================================
// DOM Elements
// =============================================================================

const elements = {
    landing: document.getElementById('landing'),
    chat: document.getElementById('chat'),
    startBtn: document.getElementById('startChat'),
    backBtn: document.getElementById('backToLanding'),
    avatar: document.getElementById('avatar'),
    chatMessages: document.getElementById('chatMessages'),
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
    typingIndicator: document.getElementById('typingIndicator'),
    quickReplies: document.getElementById('quickReplies'),
    toastContainer: document.getElementById('toastContainer')
};

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    initAvatarInteraction();
    checkApiHealth();
});

function initEventListeners() {
    // Start chat button
    elements.startBtn.addEventListener('click', startChat);
    
    // Back button
    elements.backBtn.addEventListener('click', backToLanding);
    
    // Send button
    elements.sendBtn.addEventListener('click', sendMessage);
    
    // Input handling
    elements.messageInput.addEventListener('input', handleInputChange);
    elements.messageInput.addEventListener('keydown', handleInputKeydown);
    
    // Quick replies
    document.querySelectorAll('.quick-reply').forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.dataset.question;
            elements.messageInput.value = question;
            handleInputChange();
            sendMessage();
        });
    });
}

function initAvatarInteraction() {
    const avatar = elements.avatar;
    
    // Add hover effect
    avatar.addEventListener('mouseenter', () => {
        avatar.querySelector('.mouth').style.height = '20px';
    });
    
    avatar.addEventListener('mouseleave', () => {
        avatar.querySelector('.mouth').style.height = '15px';
    });
    
    // Click to start chat
    avatar.addEventListener('click', startChat);
    
    // Eye tracking (simplified)
    document.addEventListener('mousemove', (e) => {
        const eyes = avatar.querySelectorAll('.eye');
        const avatarRect = avatar.getBoundingClientRect();
        const avatarCenterX = avatarRect.left + avatarRect.width / 2;
        const avatarCenterY = avatarRect.top + avatarRect.height / 2;
        
        const angleX = (e.clientX - avatarCenterX) / 50;
        const angleY = (e.clientY - avatarCenterY) / 50;
        
        const maxOffset = 3;
        const offsetX = Math.max(-maxOffset, Math.min(maxOffset, angleX));
        const offsetY = Math.max(-maxOffset, Math.min(maxOffset, angleY));
        
        eyes.forEach(eye => {
            eye.style.transform = `translate(${offsetX}px, ${offsetY}px)`;
        });
    });
}

// =============================================================================
// Navigation
// =============================================================================

function startChat() {
    elements.landing.classList.add('hidden');
    elements.chat.classList.remove('hidden');
    elements.messageInput.focus();
    
    // Generate new thread ID
    state.threadId = generateThreadId();
}

function backToLanding() {
    elements.chat.classList.add('hidden');
    elements.landing.classList.remove('hidden');
    
    // Reset chat state
    resetChat();
}

function resetChat() {
    state.threadId = null;
    state.messages = [];
    
    // Clear messages except welcome
    const messages = elements.chatMessages.querySelectorAll('.message:not(.welcome-message)');
    messages.forEach(msg => msg.remove());
    
    // Show quick replies again
    elements.quickReplies.classList.remove('hidden');
}

// =============================================================================
// Message Handling
// =============================================================================

function handleInputChange() {
    const value = elements.messageInput.value.trim();
    elements.sendBtn.disabled = value.length === 0;
    
    // Auto-resize textarea
    elements.messageInput.style.height = 'auto';
    elements.messageInput.style.height = Math.min(elements.messageInput.scrollHeight, 120) + 'px';
}

function handleInputKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!elements.sendBtn.disabled) {
            sendMessage();
        }
    }
}

async function sendMessage() {
    const message = elements.messageInput.value.trim();
    if (!message || state.isLoading) return;
    
    // Clear input
    elements.messageInput.value = '';
    handleInputChange();
    
    // Hide quick replies after first message
    elements.quickReplies.classList.add('hidden');
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Show typing indicator
    showTypingIndicator();
    
    // Send to API
    try {
        state.isLoading = true;
        const response = await askQuestion(message);
        
        hideTypingIndicator();
        
        // Add bot response
        addMessage(response.answer, 'bot', response.sources);
        
        // Update thread ID
        state.threadId = response.thread_id;
        
    } catch (error) {
        hideTypingIndicator();
        showToast('Erreur: ' + error.message, 'error');
        addMessage('Désolé, une erreur s\'est produite. Veuillez réessayer.', 'bot');
    } finally {
        state.isLoading = false;
    }
}

function addMessage(content, type, sources = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const time = new Date().toLocaleTimeString('fr-FR', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    
    const avatar = type === 'bot' ? '📚' : '👤';
    
    // Format content: full Markdown for bot, plain text for user
    const formattedContent = type === 'bot'
        ? formatMarkdownContent(content)
        : escapeHtml(content).replace(/\n/g, '<br>');
    
    // Format sources if available (for bot messages)
    let sourcesHTML = '';
    if (type === 'bot' && sources && sources.length > 0) {
        sourcesHTML = `
            <div class="message-sources">
                <div class="sources-header">
                    <span class="sources-icon">📄</span>
                    <span>Sources (${sources.length})</span>
                </div>
                <div class="sources-list">
                    ${sources.map((source, idx) => `
                        <div class="source-item">
                            <span class="source-number">${idx + 1}</span>
                            <div class="source-content">
                                ${source.article_number ? `<strong>Article ${source.article_number}</strong>` : ''}
                                ${source.section_title ? `<span class="source-section">${source.section_title}</span>` : ''}
                                ${source.content ? `<p class="source-text">${source.content.substring(0, 150)}${source.content.length > 150 ? '...' : ''}</p>` : ''}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-bubble">
                ${formattedContent}
                ${sourcesHTML}
            </div>
            <span class="message-time">${time}</span>
        </div>
    `;
    
    elements.chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    scrollToBottom();
}

/**
 * Renders bot message as Markdown (tables, headers, lists, bold) and sanitizes HTML.
 */
function formatMarkdownContent(content) {
    if (!content || typeof content !== 'string') return '<p></p>';
    if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
        return escapeHtml(content).replace(/\n/g, '<br>');
    }
    marked.setOptions({ breaks: true });
    const rawHtml = marked.parse(content.trim());
    const safeHtml = DOMPurify.sanitize(rawHtml, {
        ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'b', 'i', 'u', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'blockquote', 'hr', 'span', 'div'],
        ALLOWED_ATTR: ['class', 'align']
    });
    return safeHtml || '<p></p>';
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showTypingIndicator() {
    elements.typingIndicator.classList.remove('hidden');
    scrollToBottom();
}

function hideTypingIndicator() {
    elements.typingIndicator.classList.add('hidden');
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// =============================================================================
// API Communication
// =============================================================================

async function askQuestion(question) {
    const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.ASK}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            question: question,
            thread_id: state.threadId
        })
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Erreur serveur');
    }
    
    return response.json();
}

async function checkApiHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.HEALTH}`);
        if (response.ok) {
            console.log('✅ API is healthy');
        } else {
            console.warn('⚠️ API health check failed');
            showToast('L\'API n\'est pas disponible', 'error');
        }
    } catch (error) {
        console.error('❌ Cannot connect to API:', error);
        showToast('Impossible de se connecter à l\'API', 'error');
    }
}

// =============================================================================
// Utilities
// =============================================================================

function generateThreadId() {
    return 'thread_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    elements.toastContainer.appendChild(toast);
    
    // Remove after 5 seconds
    setTimeout(() => {
        toast.style.animation = 'toastIn 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// =============================================================================
// Export for testing (optional)
// =============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CONFIG, state, askQuestion };
}

