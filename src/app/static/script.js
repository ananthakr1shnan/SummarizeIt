   

document.addEventListener('DOMContentLoaded', function() {
       
    initializeApp();
});

function initializeApp() {
       
    setupEventListeners();
    
       
    updateCharacterCount();
    
       
    checkAPIHealth();
}

function setupEventListeners() {
       
    document.querySelectorAll('input[name="inputMode"]').forEach(radio => {
        radio.addEventListener('change', handleInputModeChange);
    });
    
       
    const textForm = document.getElementById('summarizeForm');
    if (textForm) {
        textForm.addEventListener('submit', handleTextSubmit);
    }
    
       
    const fileForm = document.getElementById('fileUploadForm');
    if (fileForm) {
        fileForm.addEventListener('submit', handleFileSubmit);
    }
    
       
    const inputText = document.getElementById('inputText');
    if (inputText) {
        inputText.addEventListener('input', updateCharacterCount);
    }
}

function handleInputModeChange(event) {
    const textMode = document.getElementById('textInputMode');
    const fileMode = document.getElementById('fileInputMode');
    
    if (event.target.id === 'textMode') {
        textMode.classList.remove('d-none');
        fileMode.classList.add('d-none');
    } else {
        textMode.classList.add('d-none');
        fileMode.classList.remove('d-none');
    }
}

function updateCharacterCount() {
    const inputText = document.getElementById('inputText');
    const charCount = document.getElementById('charCount');
    
    if (inputText && charCount) {
        const count = inputText.value.length;
        charCount.textContent = count.toLocaleString();
        
           
        if (count < 100) {
            charCount.className = 'text-warning';
        } else if (count < 1000) {
            charCount.className = 'text-primary';
        } else {
            charCount.className = 'text-success';
        }
    }
}

async function handleTextSubmit(event) {
    event.preventDefault();
    
    const inputText = document.getElementById('inputText').value.trim();
    const summaryLength = document.getElementById('summaryLength').value;
    const inputType = document.getElementById('inputType').value;
    
    if (!inputText) {
        showAlert('Please enter some text to summarize.', 'warning');
        return;
    }
    
    if (inputText.length < 50) {
        showAlert('Text is too short. Please enter at least 50 characters.', 'warning');
        return;
    }
    
       
    showLoading();
    
    try {
        const response = await fetch('/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: inputText,
                summary_length: summaryLength,
                input_type: inputType
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to generate summary');
        }
        
        const result = await response.json();
        displaySummaryResult(result);
        
    } catch (error) {
        console.error('Error:', error);
        showAlert(`Error: ${error.message}`, 'danger');
        hideLoading();
    }
}

async function handleFileSubmit(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const summaryLength = document.getElementById('fileSummaryLength').value;
    const inputType = document.getElementById('fileInputType').value;
    
    if (!fileInput.files[0]) {
        showAlert('Please select a file to upload.', 'warning');
        return;
    }
    
    const file = fileInput.files[0];
    
       
    if (file.size > 10 * 1024 * 1024) {
        showAlert('File size too large. Please select a file smaller than 10MB.', 'warning');
        return;
    }
    
       
    showFileLoading();
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('summary_length', summaryLength);
        formData.append('input_type', inputType);
        
        const response = await fetch('/summarize/file', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to process file');
        }
        
        const result = await response.json();
        displayFileResult(result);
        
    } catch (error) {
        console.error('Error:', error);
        showAlert(`Error: ${error.message}`, 'danger');
        hideFileLoading();
    }
}

function showLoading() {
    document.getElementById('loadingIndicator').classList.remove('d-none');
    document.getElementById('summaryResult').classList.add('d-none');
    document.getElementById('emptyState').classList.add('d-none');
    document.getElementById('summarizeBtn').disabled = true;
}

function hideLoading() {
    document.getElementById('loadingIndicator').classList.add('d-none');
    document.getElementById('emptyState').classList.remove('d-none');
    document.getElementById('summarizeBtn').disabled = false;
}

function showFileLoading() {
    const btn = document.getElementById('fileUploadBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
}

function hideFileLoading() {
    const btn = document.getElementById('fileUploadBtn');
    btn.disabled = false;
    btn.innerHTML = '<i class="fas fa-upload me-2"></i>Upload & Summarize';
}

function displaySummaryResult(result) {
       
    document.getElementById('loadingIndicator').classList.add('d-none');
    document.getElementById('emptyState').classList.add('d-none');
    
       
    document.getElementById('summaryText').textContent = result.summary;
    document.getElementById('originalLength').textContent = result.original_length.toLocaleString();
    document.getElementById('summaryLengthDisplay').textContent = result.summary_length.toLocaleString();
    document.getElementById('compressionRatio').textContent = result.compression_ratio + 'x';
    document.getElementById('detectedType').textContent = `Detected as: ${result.input_type}`;
    
       
    document.getElementById('summaryResult').classList.remove('d-none');
    document.getElementById('summarizeBtn').disabled = false;
    
       
    document.getElementById('summaryResult').scrollIntoView({
        behavior: 'smooth',
        block: 'nearest'
    });
}

function displayFileResult(result) {
    const fileResultDiv = document.getElementById('fileResult');
    
    fileResultDiv.innerHTML = `
        <div class="alert alert-success">
            <h6><i class="fas fa-check-circle me-2"></i>File Processed: ${result.filename}</h6>
            <div class="mt-3 p-3 bg-light rounded">${result.summary}</div>
        </div>
        
        <div class="row">
            <div class="col-md-4">
                <div class="card border-primary">
                    <div class="card-body text-center">
                        <h6 class="card-title">Original</h6>
                        <span class="h4 text-primary">${result.original_length.toLocaleString()}</span>
                        <small class="d-block text-muted">words</small>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card border-success">
                    <div class="card-body text-center">
                        <h6 class="card-title">Summary</h6>
                        <span class="h4 text-success">${result.summary_length.toLocaleString()}</span>
                        <small class="d-block text-muted">words</small>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card border-info">
                    <div class="card-body text-center">
                        <h6 class="card-title">Compression</h6>
                        <span class="h4 text-info">${result.compression_ratio}x</span>
                        <small class="d-block text-muted">${result.input_type}</small>
                    </div>
                </div>
            </div>
        </div>
        
        <button class="btn btn-outline-primary mt-3" onclick="copyFileResult('${result.summary}')">
            <i class="fas fa-copy me-2"></i>Copy Summary
        </button>
    `;
    
    fileResultDiv.classList.remove('d-none');
    hideFileLoading();
    
       
    fileResultDiv.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest'
    });
}

function copySummary() {
    const summaryText = document.getElementById('summaryText').textContent;
    copyToClipboard(summaryText);
}

function copyFileResult(text) {
    copyToClipboard(text);
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showAlert('Summary copied to clipboard!', 'success');
    }).catch(() => {
           
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        showAlert('Summary copied to clipboard!', 'success');
    });
}

function showAlert(message, type) {
       
    const existingAlerts = document.querySelectorAll('.alert-temporary');
    existingAlerts.forEach(alert => alert.remove());
    
       
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show alert-temporary position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 1050; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
       
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

async function checkAPIHealth() {
    try {
        const response = await fetch('/health');
        const health = await response.json();
        
        if (!health.model_loaded) {
            showAlert('Warning: Model not fully loaded. Some features may not work.', 'warning');
        }
        
        console.log('API Health:', health);
    } catch (error) {
        console.error('Health check failed:', error);
        showAlert('Warning: Unable to connect to API.', 'danger');
    }
}

   
function formatNumber(num) {
    return num.toLocaleString();
}

   
function getFileExtension(filename) {
    return filename.split('.').pop().toLowerCase();
}
