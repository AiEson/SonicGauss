// Main Three.js variables
let scene, camera, renderer, controls;
let raycaster, mouse;
let pointMap = new Map(); // To store the relationship between spheres and their IDs
let sphereMap = new Map(); // To store references to all spheres by their IDs
let sphereTimeouts = new Map(); // To track color reset timeouts for each sphere
let currentAudio = null; // To track the currently playing audio
let currentModelFile = 'data.js'; // Default model file

// Initialize the scene
function init() {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    // Get container dimensions
    const container = document.getElementById('webdemo-container');
    if (!container) {
        console.error('Webdemo container not found');
        return;
    }
    const rect = container.getBoundingClientRect();
    const containerWidth = rect.width || 800;
    const containerHeight = rect.height || 500;

    // Create camera
    camera = new THREE.PerspectiveCamera(75, containerWidth / containerHeight, 0.1, 1000);
    camera.position.z = 0.5;

    // Create renderer with proper settings for better materials
    renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: true // Enable this for better compatibility
    });
    renderer.setSize(containerWidth, containerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputEncoding = THREE.sRGBEncoding; // Important for correct color rendering
    renderer.physicallyCorrectLights = true; // Enable physically correct lighting
    document.getElementById('webdemo-container').appendChild(renderer.domElement);

    // Add orbit controls to move the camera around
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controls.screenSpacePanning = false;
    controls.maxDistance = 2;

    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Add some additional lights for better illumination
    const backLight = new THREE.DirectionalLight(0xffffff, 0.4);
    backLight.position.set(-1, -1, -1);
    scene.add(backLight);

    // Setup for ray casting (to detect clicks on spheres)
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // Add event listener for clicks
    window.addEventListener('click', onMouseClick, false);
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);

    // Start animation loop
    animate();
    
    // Display loading message
    updateStatus('Loading model... Please wait');
    
    // Automatically load the model after a short delay to allow the UI to render
    setTimeout(() => {
        try {
            // Load the model with base64 data
            loadModelFromBase64();
            // Points will be loaded after the model is loaded in the loadObjModel function
        } catch (error) {
            console.error('Error loading model:', error);
            updateStatus('Error loading model. See console for details.');
        }
    }, 100);
}

// Clear the scene of previous models
function clearScene() {
    // Get references to lights first
    const lights = [];
    scene.traverse(obj => {
        if (obj.type === 'AmbientLight' || obj.type === 'DirectionalLight') {
            lights.push(obj);
        }
    });
    
    // Clear the scene
    scene.clear();
    
    // Re-add the lights
    lights.forEach(light => scene.add(light));
    
    // Clear all sphere timeouts to prevent memory leaks
    sphereTimeouts.forEach(timeoutId => {
        clearTimeout(timeoutId);
    });
    sphereTimeouts.clear();
    
    // Clear other maps
    pointMap.clear();
    sphereMap.clear();
}

// Load the 3D model from base64 data
function loadModelFromBase64() {
    // Update loading status
    updateStatus('Processing model... Please be patient');
    
    // Load texture directly from the data URL
    let texture = null;
    if (window.modelData.textureData) {
        console.log('Loading texture from data URL');
        const textureLoader = new THREE.TextureLoader();
        
        // Create a separate image element to check loaded dimensions
        const img = new Image();
        img.onload = function() {
            console.log(`Texture dimensions: ${img.width} x ${img.height}`);
        };
        img.src = window.modelData.textureData;
        
        // Important: Direct use of the data URL prevents CORS issues
        texture = textureLoader.load(
            window.modelData.textureData, // Use the data URL directly
            // onLoad callback
            function(loadedTexture) {
                console.log('Texture loaded successfully');
                console.log(`Texture image dimensions: ${loadedTexture.image.width} x ${loadedTexture.image.height}`);
                
                // Try with different texture settings
                loadedTexture.encoding = THREE.sRGBEncoding;
                loadedTexture.flipY = true; // Try with TRUE instead of false
                loadedTexture.anisotropy = 16; // Improve texture quality
                
                // Once the texture is loaded, load the model
                loadObjWithoutFile(window.modelData.objData, texture);
            },
            // onProgress callback
            function(xhr) {
                const percent = Math.round(xhr.loaded / xhr.total * 100);
                console.log(`Texture: ${percent}% loaded`);
                updateStatus(`Loading texture: ${percent}%`);
            },
            // onError callback
            function(error) {
                console.error('Error loading texture:', error);
                // Load the model without texture if there's an error
                loadObjWithoutFile(window.modelData.objData, null);
            }
        );
    } else {
        // No texture, just load the model
        loadObjWithoutFile(window.modelData.objData, null);
    }
}

// Load OBJ data directly with a custom material using the provided texture
function loadObjWithoutFile(objDataUrl, texture) {
    // Create a material with the texture
    const material = createMaterialWithTexture(texture);
    
    // Now load the OBJ directly from string
    const objLoader = new THREE.OBJLoader();
    
    console.log('Parsing OBJ data directly...');
    
    // First get the OBJ content from the data URL
    let objContent = atob(objDataUrl.split(',')[1]);
    
    try {
        // Parse OBJ content directly
        const object = objLoader.parse(objContent);
        console.log('OBJ parsed successfully');
        
        // Apply the material to all meshes in the object
        object.traverse(function(child) {
            if (child instanceof THREE.Mesh) {
                // Keep the original geometry but update the material
                const originalGeometry = child.geometry;
                
                // Ensure geometry has UV coordinates
                if (originalGeometry.attributes.uv) {
                    console.log('Mesh has UV coordinates', child.name);
                    
                    // Log a sample of UV coordinates to debug
                    const uvAttribute = originalGeometry.attributes.uv;
                    console.log(`UV attribute: ${uvAttribute.count} coordinates`);
                    if (uvAttribute.count > 0) {
                        console.log('First few UV coordinates:');
                        for (let i = 0; i < Math.min(10, uvAttribute.count); i++) {
                            console.log(`UV[${i}]: (${uvAttribute.getX(i)}, ${uvAttribute.getY(i)})`);
                        }
                    }
                    
                    // Apply custom UV transformation if needed
                    if (false) { // Set to true to enable UV transformation
                        console.log('Applying UV transformation to fix texture mapping');
                        const uvs = originalGeometry.attributes.uv.array;
                        for (let i = 0; i < uvs.length; i += 2) {
                            // Flip Y coordinate
                            uvs[i + 1] = 1 - uvs[i + 1];
                            
                            // If needed, also flip X coordinate
                            // uvs[i] = 1 - uvs[i];
                        }
                        originalGeometry.attributes.uv.needsUpdate = true;
                    }
                } else {
                    console.warn('Mesh is missing UV coordinates:', child.name);
                }
                
                // Apply the material
                child.material = material;
                console.log('Applied material to mesh:', child.name);
            }
        });
        
        // Process and add the object to the scene
        processLoadedObject(object);
        
        // Load points after the model is loaded
        loadPoints();
        
        updateStatus('Model loaded. Click white spheres to hear sounds.');
    } catch (error) {
        console.error('Error parsing OBJ data:', error);
        updateStatus('Error loading model');
    }
}

// Create a material with the provided texture based on MTL properties
function createMaterialWithTexture(texture) {
    // Try a different material type (MeshBasicMaterial) which might handle textures differently
    const material = new THREE.MeshPhysicalMaterial({
        color: 0xffffff,            // White color (Kd 1.0 1.0 1.0)
        metalness: 0.5,             // Approximation from Ks value
        roughness: 0.1,             // Approximation from Ns value
        clearcoat: 0.3,             // Add some clearcoat for a better look
        reflectivity: 0.5,          // Add some reflectivity
        transparent: false,          // Not transparent
        map: texture                 // The diffuse texture map
    });
    
    // If we have a texture, try alternate texture settings
    if (texture) {
        // Reset texture transform
        texture.matrixAutoUpdate = false;
        const matrix = new THREE.Matrix3();
        // Try to fix the UV mapping with a transformation matrix
        // Rotate the texture if needed (in radians)
        //matrix.rotate(Math.PI); // Rotate 180 degrees if needed
        texture.matrix.copy(matrix);
        
        // Configure texture settings
        texture.encoding = THREE.sRGBEncoding;
        texture.flipY = true;  // Try with flipY true instead
        
        // Set different wrapping and filtering
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.magFilter = THREE.LinearFilter;
        texture.minFilter = THREE.LinearFilter; // No mipmaps
        texture.generateMipmaps = false;  // Disable mipmaps
        
        // Apply texture to the material
        material.map = texture;
        material.needsUpdate = true;
        
        console.log('Applied texture with special settings');
    }
    
    return material;
}

// Helper function is no longer needed as we're using data URLs directly

// These functions are no longer needed as we're using a completely different approach

// Process a loaded 3D object
function processLoadedObject(object) {
    // Center the model
    const box = new THREE.Box3().setFromObject(object);
    const center = box.getCenter(new THREE.Vector3());
    
    // Log information about the model
    console.log('Model loaded:', object);
    console.log('Materials on model:');
    object.traverse(child => {
        if (child instanceof THREE.Mesh) {
            console.log(child.name, child.material);
        }
    });
    
    // Set object position, scale, rotation as needed
    object.position.x = -center.x;
    object.position.y = -center.y;
    object.position.z = -center.z;
    
    // Add the model to the scene
    scene.add(object);
    
    // Update status message
    updateStatus('Model loaded. Click white spheres to hear sounds.');
    
    // Volume warning is now shown in instructions
}

// Load points from the pre-loaded points data and create interactive spheres
function loadPoints() {
    if (!window.pointsData) {
        console.error('Points data not available');
        return;
    }
    
    // Clear any existing maps
    pointMap.clear();
    sphereMap.clear();
    
    // Create a reusable sphere geometry
    const sphereGeometry = new THREE.SphereGeometry(0.005, 16, 16);
    
    // Iterate through the points data and create a sphere for each point
    Object.entries(window.pointsData).forEach(([id, coordinates]) => {
        // Create a completely new material instance for each sphere
        const sphereMaterial = new THREE.MeshBasicMaterial({ 
            color: 0xffffff,
            name: `material-${id}`
        });
        
        // Create a sphere at the point location with its own material
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial.clone());
        sphere.name = `sphere-${id}`;
        sphere.position.set(coordinates[0], coordinates[1], coordinates[2]);
        scene.add(sphere);
        
        // Store both the ID mapping and a reference to the sphere itself
        pointMap.set(sphere.id, id);
        sphereMap.set(id, sphere);
    });
}

// Play audio for a specific contact ID
function playAudio(contactId) {
    // Stop any currently playing audio
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
    }
    
    // Check if we have audio data for this contact
    if (window.audioData && window.audioData[contactId]) {
        console.log(`Playing audio for ${contactId}`);
        
        // Create a new audio element
        const audio = new Audio(window.audioData[contactId]);
        
        // Add event listeners
        audio.addEventListener('play', () => {
            console.log(`Audio for ${contactId} started playing`);
            updateStatus(`Playing audio for: ${contactId}...`);
        });
        
        audio.addEventListener('ended', () => {
            console.log(`Audio for ${contactId} finished playing`);
            updateStatus(`Selected Point: ${contactId}`);
            currentAudio = null;
        });
        
        audio.addEventListener('error', (e) => {
            console.error(`Error playing audio for ${contactId}:`, e);
            updateStatus(`Error playing audio for: ${contactId}`);
            currentAudio = null;
        });
        
        // Play the audio
        audio.play();
        currentAudio = audio;
    } else {
        console.warn(`No audio data available for ${contactId}`);
        updateStatus(`No audio available for: ${contactId}`);
    }
}

// Handle mouse clicks on spheres
function onMouseClick(event) {
    // Calculate mouse position in normalized device coordinates relative to container
    const container = document.getElementById('webdemo-container');
    if (!container) return;
    
    const rect = container.getBoundingClientRect();
    
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    // Update the raycaster with the mouse position and camera
    raycaster.setFromCamera(mouse, camera);
    
    // Find all objects intersecting with the ray
    const intersects = raycaster.intersectObjects(scene.children, true);
    
    // If we clicked on at least one object
    if (intersects.length > 0) {
        // Loop through the intersections to find the first sphere
        for (let i = 0; i < intersects.length; i++) {
            const object = intersects[i].object;
            
            // Check if this is one of our interactive spheres
            const pointId = pointMap.get(object.id);
            if (pointId) {
                // Get the specific sphere from our map
                const sphere = sphereMap.get(pointId);
                if (!sphere) continue;
                
                // Update the display with the selected point ID
                updateStatus(`Selected Point: ${pointId}`);
                console.log('Clicked on point:', pointId);
                
                // Store the original color of this sphere (usually white)
                const originalColor = 0xffffff;
                
                // Change only this sphere's color to red
                sphere.material.color.set(0xff0000);
                
                // Play audio for this contact point
                playAudio(pointId);
                
                // Clear any existing timeout for this sphere to prevent it getting stuck
                if (sphereTimeouts.has(pointId)) {
                    clearTimeout(sphereTimeouts.get(pointId));
                }
                
                // Reset color after a brief delay and store the timeout reference
                const timeoutId = setTimeout(() => {
                    sphere.material.color.setHex(originalColor);
                    sphereTimeouts.delete(pointId);
                }, 300);
                
                // Store the timeout ID for this sphere
                sphereTimeouts.set(pointId, timeoutId);
                
                break; // Only select the first sphere that was clicked
            }
        }
    }
}

// Handle window resize
function onWindowResize() {
    const container = document.getElementById('webdemo-container');
    if (container) {
        const rect = container.getBoundingClientRect();
        const containerWidth = rect.width || 800;
        const containerHeight = rect.height || 500;
        
        camera.aspect = containerWidth / containerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(containerWidth, containerHeight);
    }
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    controls.update(); // Only required if controls.enableDamping = true
    renderer.render(scene, camera);
}

// Update status message
function updateStatus(message) {
    const statusElement = document.getElementById('demo-status');
    if (statusElement) {
        statusElement.textContent = message;
        
        // Add loading class for loading states
        if (message.includes('Loading') || message.includes('Processing')) {
            statusElement.classList.add('loading');
        } else {
            statusElement.classList.remove('loading');
        }
        
        // Auto-hide success messages after 5 seconds
        if (message.includes('loaded') || message.includes('Selected Point')) {
            setTimeout(() => {
                if (statusElement.textContent === message) {
                    statusElement.textContent = 'Click white spheres to hear sounds';
                    statusElement.classList.remove('loading');
                }
            }, 5000);
        }
    }
}

// Remove splash screen functionality
// (handleSplashScreen function removed as it's not needed in embedded context)

// Remove volume warning functionality
// (showVolumeWarning function removed as instructions are shown directly)

// Remove splash screen call
// handleSplashScreen();

// Function to switch models
async function switchModel(modelFile) {
    try {
        // Update status
        updateStatus(`Loading model from ${modelFile}...`);
        
        // Clear the current scene
        clearScene();
        
        // Update current model file
        currentModelFile = modelFile;
        
        // Load the new model data using fetch to avoid const redeclaration
        const response = await fetch(`./static/js/webdemo/${modelFile}`);
        if (!response.ok) {
            throw new Error(`Failed to load ${modelFile}: ${response.status}`);
        }
        
        const dataContent = await response.text();
        
        // Clear existing global variables
        window.modelData = undefined;
        window.pointsData = undefined;
        window.audioData = undefined;
        
        // Execute the data file content in a way that allows reassignment
        const modifiedContent = dataContent
            .replace(/^const modelData/m, 'window.modelData')
            .replace(/^const pointsData/m, 'window.pointsData')
            .replace(/^const audioData/m, 'window.audioData');
        
        // Use Function constructor to execute the modified content
        const executeData = new Function(modifiedContent);
        executeData();
        
        // Verify data was loaded
        if (!window.modelData) {
            throw new Error('Model data not loaded properly');
        }
        
        // Load the model with base64 data
        loadModelFromBase64();
        
        // Update the active button state
        updateActiveButton(currentModelFile);
        
        return true;
    
    } catch (error) {
        console.error('Error switching models:', error);
        updateStatus(`Error loading model: ${error.message}`);
        return false;
    }
}

// Enhanced model selector initialization with buttons
function initModelSelector() {
    const modelButtons = document.querySelectorAll('.model-btn');
    
    if (modelButtons.length > 0) {
        // Add click handlers to all model buttons
        modelButtons.forEach(button => {
            button.addEventListener('click', () => {
                const selectedModel = button.dataset.model;
                if (selectedModel !== currentModelFile) {
                    switchModel(selectedModel);
                }
            });
        });
        
        // Set initial active button (ceramic bowl is default)
        updateActiveButton(currentModelFile);
    }
}

// Function to update the active button state
function updateActiveButton(activeModelFile) {
    const modelButtons = document.querySelectorAll('.model-btn');
    
    modelButtons.forEach(button => {
        if (button.dataset.model === activeModelFile) {
            button.classList.remove('is-light');
            button.classList.add('is-primary', 'current-model');
        } else {
            button.classList.remove('is-primary', 'current-model');
            button.classList.add('is-light');
        }
    });
}

// Start the application
init();

// Initialize model selector immediately
initModelSelector();

// Also try initializing after a short delay in case DOM isn't fully loaded yet
setTimeout(initModelSelector, 500);
