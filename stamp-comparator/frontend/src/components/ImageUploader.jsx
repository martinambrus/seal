import React, { useState, useRef } from 'react';

export const ImageUploader = ({ label, onImageSelected, currentImage }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [preview, setPreview] = useState(currentImage || null);
    const [fileName, setFileName] = useState('');
    const [fileSize, setFileSize] = useState('');
    const fileInputRef = useRef(null);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFile(file);
        }
    };

    const handleFileInput = (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    };

    const handleFile = (file) => {
        // Create preview URL
        const previewUrl = URL.createObjectURL(file);
        setPreview(previewUrl);
        setFileName(file.name);

        // Format file size
        const sizeInKB = (file.size / 1024).toFixed(1);
        const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);
        setFileSize(file.size > 1024 * 1024 ? `${sizeInMB} MB` : `${sizeInKB} KB`);

        // Notify parent component
        onImageSelected(file, previewUrl);
    };

    const handleClear = () => {
        setPreview(null);
        setFileName('');
        setFileSize('');
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
        onImageSelected(null, null);
    };

    const handleClick = () => {
        fileInputRef.current?.click();
    };

    return (
        <div className="image-uploader">
            <h3>{label}</h3>

            {!preview ? (
                <div
                    className={`dropzone ${isDragging ? 'dragging' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={handleClick}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/jpeg,image/png,image/jpg"
                        onChange={handleFileInput}
                        style={{ display: 'none' }}
                    />
                    <div className="dropzone-content">
                        <div className="upload-icon">📁</div>
                        <p className="dropzone-text">Drag & drop image here</p>
                        <p className="dropzone-subtext">or click to select</p>
                        <span className="file-types">JPG, PNG accepted</span>
                    </div>
                </div>
            ) : (
                <div className="preview-container">
                    <img src={preview} alt={label} className="preview-image" />
                    <div className="preview-info">
                        <div className="file-details">
                            <span className="file-name">{fileName}</span>
                            <span className="file-size">{fileSize}</span>
                        </div>
                        <button onClick={handleClear} className="clear-button">
                            ✕ Remove
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};
