import React, { useRef, useEffect, useState } from 'react';

export const ImageViewer = ({
    image,
    title,
    width = 400,
    height = 400,
    onMouseMove,
    onMouseLeave,
    overlayMask = null
}) => {
    const canvasRef = useRef(null);
    const [imageLoaded, setImageLoaded] = useState(false);
    const imgRef = useRef(null);

    useEffect(() => {
        if (!image) return;

        const img = new Image();
        img.onload = () => {
            imgRef.current = img;
            setImageLoaded(true);
            drawCanvas();
        };
        img.src = image;
    }, [image]);

    useEffect(() => {
        if (imageLoaded) {
            drawCanvas();
        }
    }, [overlayMask, imageLoaded]);

    const drawCanvas = () => {
        const canvas = canvasRef.current;
        if (!canvas || !imgRef.current) return;

        const ctx = canvas.getContext('2d');
        canvas.width = width;
        canvas.height = height;

        // Draw original image
        ctx.drawImage(imgRef.current, 0, 0, width, height);

        // Draw overlay if provided
        if (overlayMask) {
            const overlayCanvas = document.createElement('canvas');
            overlayCanvas.width = overlayMask.width;
            overlayCanvas.height = overlayMask.height;
            const overlayCtx = overlayCanvas.getContext('2d');
            overlayCtx.putImageData(overlayMask, 0, 0);

            ctx.globalAlpha = 0.5;
            ctx.drawImage(overlayCanvas, 0, 0, width, height);
            ctx.globalAlpha = 1.0;
        }
    };

    const handleMouseMove = (e) => {
        if (!onMouseMove) return;

        const rect = canvasRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Convert to image coordinates
        const imgX = (x / width) * imgRef.current.width;
        const imgY = (y / height) * imgRef.current.height;

        onMouseMove(imgX, imgY);
    };

    return (
        <div className="image-viewer">
            <h3>{title}</h3>
            <div className="canvas-container">
                <canvas
                    ref={canvasRef}
                    className="viewer-canvas"
                    onMouseMove={handleMouseMove}
                    onMouseLeave={onMouseLeave}
                />
                {!imageLoaded && (
                    <div className="loading-spinner">
                        <div className="spinner"></div>
                        <span>Loading...</span>
                    </div>
                )}
            </div>
        </div>
    );
};
