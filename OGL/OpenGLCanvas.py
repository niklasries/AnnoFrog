# ==============================================================================
# OPENGL CANVAS WIDGET
# ==============================================================================

import numpy as np
import ctypes
import cv2


# --- OpenGL Imports ---
from OpenGL.GL import *

# --- PyQt6 Imports ---
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QPointF, QRectF, QPoint

from PyQt6.QtGui import ( QImage,  QVector2D, QVector3D, QMouseEvent, QWheelEvent, 
                          QKeyEvent, QPainter, QPaintEvent, QFontMetrics, QFont, QMatrix4x4
)
from PyQt6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QWidget

from Annotations.AnnotationMode import AnnotationMode
from Annotations.AnnotationHandler import AnnotationHandler
from Annotations import PersonAnnotation
from UI import Window



MAX_ANNOTATION_VBO_VERTS = 4096  # Max vertices for the annotation Vertex Buffer Object
KEYPOINT_SCREEN_SIZE_PX = 8.0  # Default screen size of keypoints in pixels

# ==============================================================================
# ANNOTATION DATA CONSTANTS
# ==============================================================================
NUM_KEYPOINTS = 14  # Number of keypoints per person
KEYPOINT_NAMES = [
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
    "Head", "Neck"
]
SKELETON_EDGES = [  # Pairs of keypoint indices to draw skeleton lines
    (12, 13), (0, 13), (1, 13), (0, 2), (1, 3), (2, 4), (3, 5),
    (0, 6), (1, 7), (6, 8), (7, 9), (8, 10), (9, 11), (6, 7),
]

# Keypoint visibility states
VISIBILITY_NOT_SET = 0        # Keypoint not yet annotated
VISIBILITY_OCCLUDED = 1       # Keypoint annotated as occluded
VISIBILITY_VISIBLE = 2        # Keypoint annotated as visible
VISIBILITY_SUGGESTED = 3      # Keypoint is an interpolated suggestion
VISIBILITY_AI_SUGGESTED = 4   # Keypoint is an AI-generated suggestion



class OpenGLCanvas(QOpenGLWidget):
    """
    Custom QOpenGLWidget for displaying video frames and annotations.
    Handles user interaction for creating and editing annotations.
    """
    # --- Signals ---
    status_message_changed = pyqtSignal(str)        # Emits messages for the status bar
    persons_list_updated = pyqtSignal(int)          # Emits (frame_idx) when persons list needs refresh
    mode_changed_by_canvas = pyqtSignal(AnnotationMode) # Emits new mode when changed internally
    annotation_action_completed = pyqtSignal()      # Emitted after a significant annotation action (e.g., keypoint placed, bbox created)

    def __init__(self, annotation_handler: AnnotationHandler, parent: QWidget | None = None):
        super().__init__(parent)
        self.annotation_handler: AnnotationHandler = annotation_handler

        # --- Widget Properties ---
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAutoFillBackground(False) # We handle clearing in paintGL
        self.setMouseTracking(True)       # Receive mouse move events even when no button is pressed

        # --- Frame Data ---
        self.gl_texture_id: int | None = None
        self.current_frame_pixel_data = None # Raw RGB bytes for GL texture
        self.current_frame_width: int = 0
        self.current_frame_height: int = 0
        self.current_frame_display_index: int = 0

        # --- Annotation State ---
        self.current_mode: AnnotationMode = AnnotationMode.IDLE
        self.active_person: PersonAnnotation.PersonAnnotation | None = None # Person currently being edited or selected
        self.current_keypoint_idx_to_place: int = 0 # Index of keypoint to place in PLACING_KEYPOINTS mode
        self.temp_bbox_start_norm_coords: QPointF | None = None # For CREATING_BBOX_P2 mode

        # --- Interaction State ---
        self.dragged_keypoint_info: tuple[PersonAnnotation.PersonAnnotation, int] | None = None # (person_obj, kp_idx)
        self.keypoint_drag_hit_radius_norm: float = 0.005 # Normalized image space for keypoint click/drag
        self.dragged_bbox_corner_info: tuple[PersonAnnotation.PersonAnnotation, str] | None = None # (person_obj, "TL" or "BR")
        self.bbox_corner_hit_radius_norm: float = 0.0125  # Normalized image space for bbox corner click
        self.bbox_corner_draw_size_ndc: float = 0.0125    # Size of bbox corner handles in NDC

        # --- View Control ---
        self.view_scale: float = 1.0
        self.view_translation: np.ndarray = np.array([0.0, 0.0], dtype=np.float32) # NDC space
        self.is_panning: bool = False
        self.last_mouse_drag_pos: QPointF = QPointF() # Screen space, for panning delta
        self.current_mouse_screen_pos: QPoint = QPoint() # Current mouse position in widget_coords

        # --- OpenGL Resources for Text Rendering ---
        self.text_shader_program: QOpenGLShaderProgram | None = None
        self.text_vao_id: int | None = None
        self.text_vbo: QOpenGLBuffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.font_texture_id: int | None = None
        self.glyph_data: dict[str, dict] = {} # Character glyph metrics and UVs
        self.font_atlas_width: int = 0
        self.font_atlas_height: int = 0
        self.text_font_size_pt: int = 10
        self.text_font_name: str = "Arial"

        # --- OpenGL Resources for Video Rendering ---
        self.video_shader_program: QOpenGLShaderProgram | None = None
        self.video_vao_id: int | None = None
        self.video_vbo: QOpenGLBuffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer) # Fullscreen quad

        # --- OpenGL Resources for Annotation Rendering ---
        self.annotation_shader_program: QOpenGLShaderProgram | None = None
        self.annotation_vao_id: int | None = None
        self.annotation_vbo: QOpenGLBuffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)

        self._gl_initialized: bool = False

    # --- OpenGL Initialization Methods ---
    def initializeGL(self):
        """Called once when the GL context is available."""
        print("OpenGLCanvas: Initializing OpenGL...")
        self._initialize_video_renderer()
        self._initialize_annotation_renderer()
        self._initialize_text_renderer()
        glClearColor(0.1, 0.1, 0.1, 1.0) # Dark gray background
        self._gl_initialized = True
        print("OpenGLCanvas: OpenGL Initialized.")

    def _initialize_video_renderer(self):
        """Sets up shaders, VAO, VBO for rendering the video frame texture."""
        # Vertex Shader: Transforms quad vertices based on view scale/translate, passes texture coords.
        vs_src = """
            #version 330 core
            layout(location = 0) in vec2 in_Position;
            layout(location = 1) in vec2 in_TexCoord;
            out vec2 frag_TexCoord;
            uniform vec2 view_Translate; // NDC
            uniform float view_Scale;
            void main() {
                vec2 transformed_Position = (in_Position * view_Scale) + view_Translate;
                gl_Position = vec4(transformed_Position, 0.0, 1.0);
                frag_TexCoord = in_TexCoord;
            }
        """
        # Fragment Shader: Samples the video texture.
        fs_src = """
            #version 330 core
            out vec4 out_Color;
            in vec2 frag_TexCoord;
            uniform sampler2D video_TextureSampler;
            void main() {
                out_Color = texture(video_TextureSampler, frag_TexCoord);
            }
        """
        self.video_shader_program = QOpenGLShaderProgram()
        self.video_shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs_src)
        self.video_shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs_src)
        if not self.video_shader_program.link():
            print(f"Video Shader Link Error: {self.video_shader_program.log()}")

        # Fullscreen quad vertices (NDC) and texture coordinates
        # yapf: disable
        vertices = np.array([
            # positions     # texture Coords
            -1.0,  1.0,     0.0, 0.0, # Top-left
            -1.0, -1.0,     0.0, 1.0, # Bottom-left
             1.0, -1.0,     1.0, 1.0, # Bottom-right

            -1.0,  1.0,     0.0, 0.0, # Top-left
             1.0, -1.0,     1.0, 1.0, # Bottom-right
             1.0,  1.0,     1.0, 0.0  # Top-right
        ], dtype=np.float32)
        # yapf: enable

        self.video_vao_id = glGenVertexArrays(1)
        glBindVertexArray(self.video_vao_id)

        self.video_vbo.create()
        self.video_vbo.bind()
        self.video_vbo.allocate(vertices.data, vertices.nbytes)

        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * ctypes.sizeof(GLfloat), ctypes.c_void_p(0))
        # Texture coordinate attribute
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * ctypes.sizeof(GLfloat), ctypes.c_void_p(2 * ctypes.sizeof(GLfloat)))

        glBindVertexArray(0)
        self.video_vbo.release()

        # Initialize video texture
        self.gl_texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.gl_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _initialize_annotation_renderer(self):
        """Sets up shaders, VAO, VBO for rendering annotations (points, lines)."""
        # Vertex Shader: Passes through position, sets point size.
        vs_src = """
            #version 330 core
            layout(location = 0) in vec2 in_Position; // NDC
            uniform float point_render_size; // Screen pixels
            void main() {
                gl_Position = vec4(in_Position, 0.0, 1.0);
                gl_PointSize = point_render_size;
            }
        """
        # Fragment Shader: Sets color, discards for round points.
        fs_src = """
            #version 330 core
            out vec4 out_Color;
            uniform vec3 primitive_color;
            uniform bool is_Point; // True if rendering GL_POINTS
            void main() {
                if (is_Point) {
                    if (length(gl_PointCoord - vec2(0.5)) > 0.5) discard; // Make points round
                }
                out_Color = vec4(primitive_color, 1.0);
            }
        """
        self.annotation_shader_program = QOpenGLShaderProgram()
        self.annotation_shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs_src)
        self.annotation_shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs_src)
        if not self.annotation_shader_program.link():
            print(f"Annotation Shader Link Error: {self.annotation_shader_program.log()}")

        self.annotation_vao_id = glGenVertexArrays(1)
        glBindVertexArray(self.annotation_vao_id)

        self.annotation_vbo.create()
        self.annotation_vbo.bind()
        # Allocate buffer for dynamic annotation data (position: vec2)
        self.annotation_vbo.allocate(MAX_ANNOTATION_VBO_VERTS * 2 * ctypes.sizeof(GLfloat))

        glEnableVertexAttribArray(0) # Position attribute
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        glBindVertexArray(0)
        self.annotation_vbo.release()

    def _initialize_text_renderer(self):
        """Sets up shaders, VAO, VBO, and font atlas for text rendering."""
        # Vertex Shader for Text: Transforms quad vertices for each character.
        vs_src = """
            #version 330 core
            layout (location = 0) in vec4 vertex; // vec2 pos, vec2 tex
            out vec2 TexCoords;
            uniform mat4 projection_matrix; // Orthographic projection
            void main() {
                gl_Position = projection_matrix * vec4(vertex.xy, 0.0, 1.0);
                TexCoords = vertex.zw;
            }
        """
        # Fragment Shader for Text: Samples font atlas, applies color.
        fs_src = """
            #version 330 core
            in vec2 TexCoords;
            out vec4 color;
            uniform sampler2D font_texture_sampler;
            uniform vec3 text_render_color;
            void main() {
                // Use alpha from texture for text, color from uniform
                color = vec4(text_render_color, texture(font_texture_sampler, TexCoords).a);
            }
        """
        self.text_shader_program = QOpenGLShaderProgram()
        self.text_shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs_src)
        self.text_shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs_src)
        if not self.text_shader_program.link():
            print(f"Text Shader Link Error: {self.text_shader_program.log()}")
            return

        self.text_vao_id = glGenVertexArrays(1)
        glBindVertexArray(self.text_vao_id)

        self.text_vbo.create()
        self.text_vbo.bind()
        # Allocate buffer for text quads (e.g., 50 chars, 6 verts/char, 4 floats/vert)
        self.text_vbo.allocate(50 * 6 * 4 * ctypes.sizeof(GLfloat))

        # Vertex attribute: vec4 (posX, posY, texU, texV)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * ctypes.sizeof(GLfloat), ctypes.c_void_p(0))

        glBindBuffer(GL_ARRAY_BUFFER, 0) # Unbind VBO before VAO
        glBindVertexArray(0) # Unbind VAO

        self._create_font_texture_atlas()

    def _create_font_texture_atlas(self):
        """Creates a texture atlas for characters to be rendered."""
        font = QFont(self.text_font_name, self.text_font_size_pt)
        font_metrics = QFontMetrics(font)
        # Printable ASCII characters
        char_set = "".join([chr(i) for i in range(32, 127)])
        max_char_height = font_metrics.height()
        total_width = 0
        char_details_list = []

        # 1. Get metrics for each character
        for char_code in char_set:
            bounding_rect = font_metrics.boundingRect(char_code)
            advance_width = font_metrics.horizontalAdvance(char_code)
            # Create small QImage for this char
            # Ensure width/height are at least 1 to avoid QPainter issues
            img_width = max(1, bounding_rect.width())
            img_height = max(1, bounding_rect.height())
            char_img = QImage(img_width, img_height, QImage.Format.Format_ARGB32_Premultiplied)
            char_img.fill(Qt.GlobalColor.transparent)
            painter = QPainter(char_img)
            painter.setFont(font)
            painter.setPen(Qt.GlobalColor.white) # Render white, colorize in shader
            # Adjust draw position for characters like 'g', 'p' that extend below baseline
            painter.drawText(QPoint(-bounding_rect.left(), -bounding_rect.top()), char_code)
            painter.end()

            char_details_list.append({
                'char': char_code, 'image': char_img,
                'width_px': char_img.width(), 'height_px': char_img.height(),
                'advance_px': advance_width,
                'bearing_x_px': bounding_rect.left(),
                'bearing_y_px': bounding_rect.top() # This is QPainter's bearing, relative to baseline
            })
            total_width += char_img.width() + 2 # Add padding between chars

        # 2. Create the atlas QImage
        self.font_atlas_width = total_width
        self.font_atlas_height = max_char_height + 2 # Add padding
        atlas_image = QImage(self.font_atlas_width, self.font_atlas_height, QImage.Format.Format_ARGB32_Premultiplied)
        atlas_image.fill(Qt.GlobalColor.transparent)
        atlas_painter = QPainter(atlas_image)
        current_x = 1 # Start with 1px padding

        # 3. Draw characters onto the atlas and store UV/metric data
        for details in char_details_list:
            char_img = details['image']
            atlas_painter.drawImage(QPoint(current_x, 1), char_img) # 1px top padding

            # Calculate UV coordinates (normalized)
            uv_x = current_x / self.font_atlas_width
            uv_y = 1 / self.font_atlas_height # Y is often flipped in tex coords
            uv_w = char_img.width() / self.font_atlas_width
            uv_h = char_img.height() / self.font_atlas_height

            self.glyph_data[details['char']] = {
                'uv_rect': QRectF(uv_x, uv_y, uv_w, uv_h), # Texture coordinates
                'size_px': QSize(char_img.width(), char_img.height()), # Size on atlas
                'advance_px': details['advance_px'], # How much to advance X cursor
                'bearing_px': QPointF(details['bearing_x_px'], font_metrics.boundingRect(details['char']).top()) # Bearing from origin (top-left of char)
            }
            current_x += char_img.width() + 1 # 1px padding
        atlas_painter.end()

        # 4. Create OpenGL texture from the atlas QImage
        if self.font_texture_id is None:
            self.font_texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.font_texture_id)

        if atlas_image.format() != QImage.Format.Format_ARGB32_Premultiplied:
            atlas_image = atlas_image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)

        ptr = atlas_image.constBits()
        ptr.setsize(atlas_image.sizeInBytes()) # Required for PySide/PyQt
        image_data_bytes = bytes(ptr)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.font_atlas_width, self.font_atlas_height,
                     0, GL_BGRA, GL_UNSIGNED_BYTE, image_data_bytes) # QImage BGRA -> GL_BGRA

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glBindTexture(GL_TEXTURE_2D, 0)
        print(f"Font atlas created: {self.font_atlas_width}x{self.font_atlas_height} for {len(self.glyph_data)} characters.")

    # --- OpenGL Rendering Methods ---
    def paintGL(self):
        """Called by Qt to repaint the widget. Delegates to actualPaintGL."""
        # This is the standard Qt entry point.
        # We use actualPaintGL to allow calling paint logic internally if needed,
        # though it's generally good practice for paintGL to be the sole trigger.
        self.actualPaintGL()

    def actualPaintGL(self):
        """Performs the actual OpenGL rendering commands."""
        if not self._gl_initialized: return

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1) # For tightly packed RGB data
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # Clear buffers

        # 1. Render Video Frame
        if self.video_shader_program and self.current_frame_pixel_data and self.gl_texture_id:
            self.video_shader_program.bind()
            self.video_shader_program.setUniformValue("view_Translate", QVector2D(self.view_translation[0], self.view_translation[1]))
            self.video_shader_program.setUniformValue("view_Scale", self.view_scale)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.gl_texture_id)
            # Upload current frame data to texture
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, self.current_frame_width, self.current_frame_height,
                         0, GL_RGB, GL_UNSIGNED_BYTE, self.current_frame_pixel_data)
            self.video_shader_program.setUniformValue("video_TextureSampler", 0) # Texture unit 0

            glBindVertexArray(self.video_vao_id)
            glDrawArrays(GL_TRIANGLES, 0, 6) # Draw the quad

            glBindVertexArray(0)
            glBindTexture(GL_TEXTURE_2D, 0)
            self.video_shader_program.release()

        # 2. Render Annotations (Keypoints, Skeletons, BBoxes)
        glDisable(GL_DEPTH_TEST) # Annotations on top
        self._render_annotations()

        # 3. Render Text (e.g., current keypoint name near cursor)
        if self.current_mode == AnnotationMode.PLACING_KEYPOINTS and \
           self.active_person and \
           not self.active_person.is_suggestion_any_type() and \
           not self.annotation_handler.is_person_done(self.active_person.id):

            text_to_render = ""
            if self.dragged_keypoint_info and self.dragged_keypoint_info[0] is self.active_person:
                text_to_render = f"Dragging: {KEYPOINT_NAMES[self.dragged_keypoint_info[1]]}"
            elif self.active_person.all_keypoints_set(): #
                text_to_render = ""
            else:
                if 0 <= self.current_keypoint_idx_to_place < NUM_KEYPOINTS:
                    text_to_render = f"Place: {KEYPOINT_NAMES[self.current_keypoint_idx_to_place]}"
                else:
                    next_unset_for_tooltip = self.active_person.get_next_unset_keypoint_idx(0)
                    if next_unset_for_tooltip is not None:
                        text_to_render = f"Place: {KEYPOINT_NAMES[next_unset_for_tooltip]}"
                    else:
                        text_to_render = "" # Generic

            # Calculate text position near cursor (screen coordinates)
            font = QFont(self.text_font_name, self.text_font_size_pt)
            fm = QFontMetrics(font)
            text_width_px = sum(self.glyph_data.get(char, {'advance_px': fm.horizontalAdvance(' ')}).get('advance_px', fm.horizontalAdvance(' ')) for char in text_to_render)

            offset_x_px, offset_y_px = 15, -12 # Offset from mouse cursor
            text_pos_x_px = self.current_mouse_screen_pos.x() + offset_x_px
            # Y for text rendering is usually baseline; adjust for ascent
            text_pos_y_px = self.current_mouse_screen_pos.y() + offset_y_px + fm.ascent()
            text_height_px = fm.height()

            # Keep text within canvas bounds
            if text_pos_x_px + text_width_px + 6 > self.width(): # +6 for padding
                text_pos_x_px = self.current_mouse_screen_pos.x() - offset_x_px - text_width_px
            if text_pos_y_px - fm.ascent() < 3: # Too close to top
                text_pos_y_px = self.current_mouse_screen_pos.y() + abs(offset_y_px) + text_height_px + fm.ascent()
            if text_pos_x_px < 3: text_pos_x_px = 3.0
            if text_pos_y_px + fm.descent() > self.height() - 3: # Too close to bottom
                text_pos_y_px = self.height() - fm.descent() - 3
            if text_pos_y_px - fm.ascent() < 3: # Re-check after Y adjustment
                 text_pos_y_px = float(fm.ascent() + 3)


            self._render_gl_text(text_to_render, text_pos_x_px, text_pos_y_px, QVector3D(1.0, 1.0, 1.0)) # White text

    def _render_gl_text(self, text_content: str, screen_x: float, screen_y_baseline: float, color: QVector3D):
        """Renders a string of text at specified screen coordinates using the font atlas."""
        if not all([self.text_shader_program, self.font_texture_id, self.glyph_data,
                    self.text_vbo.isCreated(), self.width() > 0, self.height() > 0]):
            return # Not ready to render text

        self.text_shader_program.bind()
        glBindVertexArray(self.text_vao_id)
        self.text_vbo.bind()

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.font_texture_id)
        self.text_shader_program.setUniformValue("font_texture_sampler", 0) # Texture unit 0
        self.text_shader_program.setUniformValue("text_render_color", color)

        proj_matrix = QMatrix4x4(
            2.0 / self.width(), 0.0,                  0.0, -1.0,  # col 1
            0.0,                -2.0 / self.height(), 0.0,  1.0,  # col 2 (Y is inverted)
            0.0,                0.0,                 -1.0,  0.0,  # col 3 (depth not used)
            0.0,                0.0,                  0.0,  1.0   # col 4
        )
        self.text_shader_program.setUniformValue("projection_matrix", proj_matrix)

        vertex_data_list = []
        current_pen_x = screen_x
        font = QFont(self.text_font_name, self.text_font_size_pt) # For fallback metrics
        font_metrics = QFontMetrics(font)

        for char_code in text_content:
            glyph = self.glyph_data.get(char_code)
            if not glyph: # Fallback for characters not in atlas (e.g., space)
                current_pen_x += font_metrics.horizontalAdvance(' ') # Use space width as fallback
                continue

            # Position of the quad for this character on screen
            x_pos = current_pen_x + glyph['bearing_px'].x()
            y_pos = screen_y_baseline + glyph['bearing_px'].y() # Bearing Y is from baseline to top of glyph
            char_width_px = glyph['size_px'].width()
            char_height_px = glyph['size_px'].height()

            # UV coordinates from atlas
            uv_rect = glyph['uv_rect']
            u, v, uw, vh = uv_rect.x(), uv_rect.y(), uv_rect.width(), uv_rect.height()

            # Quad vertices: (posX, posY, texU, texV)
            # yapf: disable
            vertex_data_list.extend([
                x_pos,                y_pos,                 u,      v,      # Top-left
                x_pos,                y_pos + char_height_px,u,      v + vh, # Bottom-left
                x_pos + char_width_px,y_pos + char_height_px,u + uw,  v + vh, # Bottom-right

                x_pos,                y_pos,                 u,      v,      # Top-left
                x_pos + char_width_px,y_pos + char_height_px,u + uw,  v + vh, # Bottom-right
                x_pos + char_width_px,y_pos,                 u + uw,  v       # Top-right
            ])
            # yapf: enable
            current_pen_x += glyph['advance_px'] # Move pen for next character

        if vertex_data_list:
            vertex_data_np = np.array(vertex_data_list, dtype=np.float32)
            self.text_vbo.write(0, vertex_data_np.data, vertex_data_np.nbytes)

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) # Standard alpha blending
            glDrawArrays(GL_TRIANGLES, 0, len(vertex_data_list) // 4) # 4 floats per vertex
            glDisable(GL_BLEND)

        glBindTexture(GL_TEXTURE_2D, 0)
        self.text_vbo.release()
        glBindVertexArray(0)
        self.text_shader_program.release()



    def _prepare_annotation_vbo_data(self) -> tuple[np.ndarray | None, list[dict]]:
        """
        Prepares vertex data and draw commands for all annotations on the current frame.
        This version explicitly handles rendering AI keypoints within existing user bounding boxes.
        """
        vertex_data_accumulator = []
        draw_commands = []
        current_vbo_offset = 0

        # --- add_render_command helper function (no changes needed here) ---
        def add_render_command(points_ndc: list[QPointF], color_vec3: QVector3D,
                               primitive_type, point_size_px: float = KEYPOINT_SCREEN_SIZE_PX,
                               is_suggestion_render_style: bool = False): # Renamed for clarity
            nonlocal current_vbo_offset
            if not points_ndc: return
            final_color = color_vec3
            final_point_size = point_size_px
            if is_suggestion_render_style: # Use this flag for styling
                if primitive_type == GL_POINTS:
                    final_color = QVector3D(color_vec3.x() * 0.6, color_vec3.y() * 0.6, color_vec3.z() * 0.6)
                    final_point_size = point_size_px * 0.8
                elif primitive_type == GL_LINES:
                    final_color = QVector3D(color_vec3.x() * 0.5, color_vec3.y() * 0.5, color_vec3.z() * 0.5)
            if not isinstance(final_point_size, float): final_point_size = float(final_point_size)
            num_vertices_added = 0
            for p_ndc in points_ndc:
                vertex_data_accumulator.extend([p_ndc.x(), p_ndc.y()])
            num_vertices_added = len(points_ndc)
            if num_vertices_added > 0:
                draw_commands.append({
                    'color': final_color, 'first': current_vbo_offset, 'count': num_vertices_added,
                    'primitive': primitive_type, 'point_size': final_point_size
                })
                current_vbo_offset += num_vertices_added
        # --- End of add_render_command helper ---

        # List of dictionaries, each representing a "render task"
        # Keys: 'person_obj', 'draw_bbox', 'draw_keypoints', 'draw_skeleton', 'is_suggestion_style'
        render_tasks = []

        current_real_annotations = self.annotation_handler.get_annotations_for_frame(self.current_frame_display_index)
        current_suggested_annotations = self.annotation_handler.get_suggested_annotations_for_frame(self.current_frame_display_index)

        
        processed_real_ids = set()
        for person_real in current_real_annotations:
            if self._is_person_globally_hidden(person_real.id):
                continue
            render_tasks.append({
                "person_obj": person_real,
                "draw_bbox": True,
                "draw_keypoints": True, # Will only draw non-suggested KPs from this object
                "draw_skeleton": True,  # Will only draw skeleton from non-suggested KPs
                "is_suggestion_style": False # Rendered as a "real" entity
            })
            processed_real_ids.add(person_real.id)

        # Step 2: Process SUGGESTED annotations.
        # These can either be "pure" suggestions (new ID) or AI keypoint overlays for existing real IDs.
        for person_sugg in current_suggested_annotations:
            if self._is_person_globally_hidden(person_sugg.id):
                continue

            is_ai_suggestion = person_sugg.has_ai_suggestions()
            is_interp_suggestion = person_sugg.has_suggestions() and not is_ai_suggestion

            # Check if a real annotation with the same ID was already added to render_tasks
            # This means there's a user-defined bounding box for this ID.
            real_task_exists_for_id = person_sugg.id in processed_real_ids

            if real_task_exists_for_id:
                if is_ai_suggestion:
                    # This AI suggestion is for an existing real person.
                    # We want to draw its keypoints and skeleton if the real person's annotation is "empty".
                    real_obj_for_id = next((p for p in current_real_annotations if p.id == person_sugg.id), None)
                    if real_obj_for_id: # Should always be found if real_task_exists_for_id
                        num_real_kps_user = sum(1 for kp_u in real_obj_for_id.keypoints if kp_u[2] in [VISIBILITY_VISIBLE, VISIBILITY_OCCLUDED])
                        if num_real_kps_user <= 1: # If the real annotation is "empty" of keypoints
                            render_tasks.append({
                                "person_obj": person_sugg,      # Use the suggestion object for its AI keypoints
                                "draw_bbox": False,             # Bbox already drawn by the real task
                                "draw_keypoints": True,
                                "draw_skeleton": True,
                                "is_suggestion_style": True     # Keypoints/skeleton styled as suggestion
                            })

            else:
                # No real annotation for this ID. This is a "pure" new suggestion.
                # Draw its bbox (if any), keypoints, and skeleton.
                render_tasks.append({
                    "person_obj": person_sugg,
                    "draw_bbox": True, # Draw the suggestion's own bbox
                    "draw_keypoints": True,
                    "draw_skeleton": True,
                    "is_suggestion_style": True # All parts styled as suggestion
                })

        # --- Drawing Loop: Iterate through the prepared render_tasks ---
        for task in render_tasks:
            person = task["person_obj"]
            is_styled_as_suggestion = task["is_suggestion_style"] # For dimming/sizing in add_render_command

            is_person_globally_done = self.annotation_handler.is_person_done(person.id)
            is_active_on_canvas = (self.active_person is not None and
                                   self.active_person.id == person.id)
            
            # Refine active_on_canvas based on whether the canvas's active_person matches the style of this task
            if self.active_person:
                if is_styled_as_suggestion: # This task is for rendering something in suggestion style
                    is_active_on_canvas = (self.active_person.id == person.id and
                                           self.active_person.is_suggestion_any_type())
                else: # This task is for rendering something in real style
                    is_active_on_canvas = (self.active_person.id == person.id and
                                           not self.active_person.is_suggestion_any_type())

            # --- Bounding Box ---
            if task["draw_bbox"] and person.bbox != [0.0, 0.0, 0.0, 0.0]:
                tl_img_norm_bbox = QPointF(min(person.bbox[0], person.bbox[2]), min(person.bbox[1], person.bbox[3]))
                br_img_norm_bbox = QPointF(max(person.bbox[0], person.bbox[2]), max(person.bbox[1], person.bbox[3]))
                tl_ndc_bbox = self._normalized_image_to_render_ndc(tl_img_norm_bbox)
                br_ndc_bbox = self._normalized_image_to_render_ndc(br_img_norm_bbox)
                tr_ndc_bbox = QPointF(br_ndc_bbox.x(), tl_ndc_bbox.y())
                bl_ndc_bbox = QPointF(tl_ndc_bbox.x(), br_ndc_bbox.y())

                bbox_color = QVector3D(0.0, 1.0, 0.0) # Default green for real, or base for suggestion
                if is_active_on_canvas and not is_styled_as_suggestion:
                    bbox_color = QVector3D(1.0, 1.0, 0.0) # Yellow for active real

                if is_person_globally_done and not is_styled_as_suggestion:
                    bbox_color.setX(bbox_color.x() * 0.3); bbox_color.setY(bbox_color.y() * 0.3); bbox_color.setZ(bbox_color.z() * 0.3)

                add_render_command([tl_ndc_bbox, tr_ndc_bbox, tr_ndc_bbox, br_ndc_bbox,
                                    br_ndc_bbox, bl_ndc_bbox, bl_ndc_bbox, tl_ndc_bbox],
                                   bbox_color, GL_LINES,
                                   is_suggestion_render_style=is_styled_as_suggestion)

                # BBox resize handles (only for active, REAL, non-done, non-hidden)
                if is_active_on_canvas and not is_styled_as_suggestion and not is_person_globally_done and \
                   not self._is_person_globally_hidden(person.id) and \
                   (self.current_mode == AnnotationMode.IDLE or self.current_mode == AnnotationMode.PLACING_KEYPOINTS):
                    corner_color = QVector3D(1.0, 0.5, 0.0)
                    handle_size_ndc_half = self.bbox_corner_draw_size_ndc / 2.0
                    # TL handle
                    tl_h_tl = QPointF(tl_ndc_bbox.x() - handle_size_ndc_half, tl_ndc_bbox.y() + handle_size_ndc_half)
                    tl_h_br = QPointF(tl_ndc_bbox.x() + handle_size_ndc_half, tl_ndc_bbox.y() - handle_size_ndc_half)
                    add_render_command([QPointF(tl_h_tl.x(), tl_h_tl.y()), QPointF(tl_h_br.x(), tl_h_tl.y()), QPointF(tl_h_br.x(), tl_h_tl.y()), QPointF(tl_h_br.x(), tl_h_br.y()), QPointF(tl_h_br.x(), tl_h_br.y()), QPointF(tl_h_tl.x(), tl_h_br.y()), QPointF(tl_h_tl.x(), tl_h_br.y()), QPointF(tl_h_tl.x(), tl_h_tl.y())], corner_color, GL_LINES, is_suggestion_render_style=False)
                    # BR handle
                    br_h_tl = QPointF(br_ndc_bbox.x() - handle_size_ndc_half, br_ndc_bbox.y() + handle_size_ndc_half)
                    br_h_br = QPointF(br_ndc_bbox.x() + handle_size_ndc_half, br_ndc_bbox.y() - handle_size_ndc_half)
                    add_render_command([QPointF(br_h_tl.x(), br_h_tl.y()), QPointF(br_h_br.x(), br_h_tl.y()), QPointF(br_h_br.x(), br_h_tl.y()), QPointF(br_h_br.x(), br_h_br.y()), QPointF(br_h_br.x(), br_h_br.y()), QPointF(br_h_tl.x(), br_h_br.y()), QPointF(br_h_tl.x(), br_h_br.y()), QPointF(br_h_tl.x(), br_h_tl.y())], corner_color, GL_LINES, is_suggestion_render_style=False)

            # --- Keypoints and Skeleton ---
            keypoints_ndc_cache = [None] * NUM_KEYPOINTS # Must be populated for skeleton even if KPs aren't drawn by this task
            temp_keypoint_params_for_skeleton = [] # To get NDC coords for skeleton

            # First pass: get all KP NDC coords for this person object for skeleton drawing
            for i, (x_norm, y_norm, visibility) in enumerate(person.keypoints):
                if visibility != VISIBILITY_NOT_SET:
                    keypoints_ndc_cache[i] = self._normalized_image_to_render_ndc(QPointF(x_norm, y_norm))

            if task["draw_keypoints"]:
                keypoint_render_params = []
                for i, (x_norm, y_norm, visibility) in enumerate(person.keypoints):
                    if visibility == VISIBILITY_NOT_SET: continue
                    # kp_ndc already in keypoints_ndc_cache[i] if set
                    kp_ndc = keypoints_ndc_cache[i]
                    if kp_ndc is None: continue # Should not happen if visibility != NOT_SET

                    is_current_kp_being_placed_or_dragged = (
                        is_active_on_canvas and not is_styled_as_suggestion and
                        not is_person_globally_done and not self._is_person_globally_hidden(person.id) and
                        ((self.current_mode == AnnotationMode.PLACING_KEYPOINTS and
                          i == self.current_keypoint_idx_to_place and not self.dragged_keypoint_info) or
                         (self.dragged_keypoint_info and
                          self.dragged_keypoint_info[0].id == person.id and
                          self.dragged_keypoint_info[1] == i))
                    )

                    kp_color_tuple = (0.5, 0.5, 0.5)
                    if visibility == VISIBILITY_SUGGESTED: kp_color_tuple = (0.6, 0.6, 0.6)
                    elif visibility == VISIBILITY_AI_SUGGESTED: kp_color_tuple = (0.3, 0.7, 0.7)
                    elif is_current_kp_being_placed_or_dragged: kp_color_tuple = (1.0, 0.5, 0.0)
                    elif visibility == VISIBILITY_VISIBLE: kp_color_tuple = (1.0, 0.0, 0.0)
                    elif visibility == VISIBILITY_OCCLUDED: kp_color_tuple = (0.0, 0.0, 1.0)

                    final_kp_color = QVector3D(*kp_color_tuple)
                    if is_person_globally_done and not is_styled_as_suggestion: # Dim done real KPs
                        final_kp_color.setX(final_kp_color.x() * 0.3); final_kp_color.setY(final_kp_color.y() * 0.3); final_kp_color.setZ(final_kp_color.z() * 0.3)

                    point_draw_size = KEYPOINT_SCREEN_SIZE_PX
                    if not is_styled_as_suggestion: # Real KPs styling
                        if is_current_kp_being_placed_or_dragged: point_draw_size *= 1.5
                        elif is_active_on_canvas and not is_person_globally_done: point_draw_size *= 1.2
                    
                    # This flag determines if the individual KP point uses suggestion styling
                    is_this_kp_point_styled_as_suggestion = visibility in [VISIBILITY_SUGGESTED, VISIBILITY_AI_SUGGESTED]
                    keypoint_render_params.append((kp_ndc, final_kp_color, point_draw_size, is_this_kp_point_styled_as_suggestion))

                keypoint_render_params.sort(key=lambda x: (x[3], id(x[1]), x[2])) # Sort to draw suggestions smaller/behind
                for kp_ndc_pos, kp_col, kp_sz, is_kp_sugg_style_flag in keypoint_render_params:
                    add_render_command([kp_ndc_pos], kp_col, GL_POINTS, point_size_px=kp_sz, is_suggestion_render_style=is_kp_sugg_style_flag)

            if task["draw_skeleton"]:
                skeleton_color = QVector3D(0.7, 0.7, 0.7) # Default gray
                if is_styled_as_suggestion: # Skeleton from a suggestion object
                    if any(kp[2] == VISIBILITY_AI_SUGGESTED for kp in person.keypoints): skeleton_color = QVector3D(0.4, 0.6, 0.6)
                elif is_active_on_canvas: # Active real person's skeleton
                    skeleton_color = QVector3D(1.0, 1.0, 0.5)
                else: # Inactive real person's skeleton
                    skeleton_color = QVector3D(1.0, 1.0, 1.0)

                if is_person_globally_done and not is_styled_as_suggestion: # Dim done real skeleton
                    skeleton_color.setX(skeleton_color.x() * 0.3); skeleton_color.setY(skeleton_color.y() * 0.3); skeleton_color.setZ(skeleton_color.z() * 0.3)

                skeleton_lines_ndc = []
                for i1, i2 in SKELETON_EDGES:
                    v1_vis = person.keypoints[i1][2]
                    v2_vis = person.keypoints[i2][2]
                    # Check if both KPs forming the edge are "set" (not VISIBILITY_NOT_SET)
                    if v1_vis != VISIBILITY_NOT_SET and v2_vis != VISIBILITY_NOT_SET and \
                       keypoints_ndc_cache[i1] and keypoints_ndc_cache[i2]:
                        skeleton_lines_ndc.extend([keypoints_ndc_cache[i1], keypoints_ndc_cache[i2]])
                
                add_render_command(skeleton_lines_ndc, skeleton_color, GL_LINES, is_suggestion_render_style=is_styled_as_suggestion)


        # --- Draw temporary elements (crosshairs, temp bbox being created) ---
        if self.current_mode == AnnotationMode.CREATING_BBOX_P2 and self.temp_bbox_start_norm_coords:
            p1_norm_img = self.temp_bbox_start_norm_coords
            p2_norm_img = self._widget_to_normalized_image_coords(self.current_mouse_screen_pos)
            p1_ndc = self._normalized_image_to_render_ndc(p1_norm_img)
            p2_ndc = self._normalized_image_to_render_ndc(p2_norm_img)
            tl_ndc = QPointF(min(p1_ndc.x(), p2_ndc.x()), min(p1_ndc.y(), p2_ndc.y()))
            br_ndc = QPointF(max(p1_ndc.x(), p2_ndc.x()), max(p1_ndc.y(), p2_ndc.y()))
            tr_ndc = QPointF(br_ndc.x(), tl_ndc.y())
            bl_ndc = QPointF(tl_ndc.x(), br_ndc.y())
            add_render_command([tl_ndc, tr_ndc, tr_ndc, br_ndc, br_ndc, bl_ndc, bl_ndc, tl_ndc],
                               QVector3D(0.5, 0.5, 1.0), GL_LINES, is_suggestion_render_style=False)

        if self.current_mode in [AnnotationMode.CREATING_BBOX_P1, AnnotationMode.CREATING_BBOX_P2] and \
           self.width() > 0 and self.height() > 0:
            mouse_ndc_canvas = self._screen_pos_to_canvas_ndc(self.current_mouse_screen_pos)
            crosshair_color = QVector3D(0.7, 0.7, 0.7)
            add_render_command([QPointF(-1.0, mouse_ndc_canvas.y()), QPointF(1.0, mouse_ndc_canvas.y())], crosshair_color, GL_LINES, is_suggestion_render_style=False)
            add_render_command([QPointF(mouse_ndc_canvas.x(), -1.0), QPointF(mouse_ndc_canvas.x(), 1.0)], crosshair_color, GL_LINES, is_suggestion_render_style=False)

        return (np.array(vertex_data_accumulator, dtype=np.float32) if vertex_data_accumulator else None, draw_commands)

    def _render_annotations(self):
        """Uploads annotation VBO data and executes draw commands."""
        if not self.annotation_shader_program or self.annotation_vao_id is None: return

        vertex_data_np, draw_commands_list = self._prepare_annotation_vbo_data()

        if vertex_data_np is None or not draw_commands_list:
            return # Nothing to render

        self.annotation_shader_program.bind()
        glBindVertexArray(self.annotation_vao_id)
        self.annotation_vbo.bind()

        # Upload data to VBO
        if vertex_data_np.nbytes > self.annotation_vbo.size():
            # Reallocate if data is larger than current VBO size
            self.annotation_vbo.allocate(vertex_data_np.data, vertex_data_np.nbytes)
        elif vertex_data_np.nbytes > 0:
            self.annotation_vbo.write(0, vertex_data_np.data, vertex_data_np.nbytes)
        else: # No data to write
            self.annotation_vbo.release()
            glBindVertexArray(0)
            self.annotation_shader_program.release()
            return

        glEnable(GL_PROGRAM_POINT_SIZE) # Allow shader to control point size
        glLineWidth(1.0) # For skeleton lines

        # Execute draw commands
        for cmd in draw_commands_list:
            self.annotation_shader_program.setUniformValue("primitive_color", cmd['color'])
            is_point_primitive = (cmd['primitive'] == GL_POINTS)
            self.annotation_shader_program.setUniformValue("point_render_size", float(cmd['point_size']) if is_point_primitive else 1.0)
            self.annotation_shader_program.setUniformValue("is_Point", is_point_primitive)
            glDrawArrays(cmd['primitive'], cmd['first'], cmd['count'])

        glDisable(GL_PROGRAM_POINT_SIZE)
        self.annotation_vbo.release()
        glBindVertexArray(0)
        self.annotation_shader_program.release()

    # --- Qt Events Overridden for QOpenGLWidget ---
    def paintEvent(self, event: QPaintEvent):
        """
        Overrides QWidget.paintEvent.
        QOpenGLWidget handles the context and calls paintGL.
        We don't need to do QPainter stuff here if all rendering is GL.
        """
        super().paintEvent(event) # Important for QOpenGLWidget's internal management

    # --- Frame and Mode Management ---
    def update_video_frame(self, frame_bgr: np.ndarray | None, frame_idx: int):
        """
        Updates the canvas with a new video frame.
        Args:
            frame_bgr: The new frame in BGR format, or None to clear.
            frame_idx: The index of the new frame.
        """
        self.current_frame_display_index = frame_idx
        if frame_bgr is None:
            self.current_frame_pixel_data = None
            self.current_frame_height, self.current_frame_width = (0, 0)
            self.update() # Request repaint
            return

        # Convert BGR to RGB for OpenGL texture
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.current_frame_height, self.current_frame_width = frame_rgb.shape[:2]

        # Ensure data is C-contiguous for tobytes()
        if not frame_rgb.flags['C_CONTIGUOUS']:
            frame_rgb = np.ascontiguousarray(frame_rgb, dtype=frame_rgb.dtype)

        self.current_frame_pixel_data = frame_rgb.tobytes()
        self.update() # Request repaint to show the new frame

    def set_annotation_mode(self, mode: AnnotationMode, active_person_candidate: PersonAnnotation.PersonAnnotation | None = None):
        """
        Sets the current annotation mode and updates UI accordingly.
        Args:
            mode: The new AnnotationMode.
            active_person_candidate: A PersonAnnotation to potentially set as active,
                                     depending on the mode.
        """
        if self.current_mode == mode and self.active_person is active_person_candidate:
            return # No change

        self.current_mode = mode
        status_text = f"Mode: {mode.name}"

        if mode == AnnotationMode.IDLE:
            # Keep current active person if one is provided and valid, or if one was already active
            self.active_person = active_person_candidate if active_person_candidate is not None else self.active_person
            active_person_text = "None"
            if self.active_person:
                active_person_text = f"P{self.active_person.id}"
                if self.active_person.is_suggestion_any_type():
                    active_person_text += " (Suggestion Shell)" # Active person is a temporary shell for a suggestion
                elif self.annotation_handler.is_person_done(self.active_person.id):
                    active_person_text += " (Done)"
            status_text = f"Mode: Idle | Frame {self.current_frame_display_index + 1} | Active: {active_person_text}"

        elif mode == AnnotationMode.CREATING_BBOX_P1:
            self.active_person = None # No active person when starting a new bbox
            self.temp_bbox_start_norm_coords = None
            status_text = "Bounding Box: Click to set Top-Left corner."

        elif mode == AnnotationMode.CREATING_BBOX_P2:
            status_text = "Bounding Box: Click to set Bottom-Right corner."

        elif mode == AnnotationMode.PLACING_KEYPOINTS:
            self.active_person = active_person_candidate # This should be a valid, non-suggestion person
            if self.active_person and not self.active_person.is_suggestion_any_type() and \
               not self.annotation_handler.is_person_done(self.active_person.id) and \
                not self._is_person_globally_hidden(self.active_person.id):
                if not self.dragged_keypoint_info: # Don't reset if dragging
                    next_unset_kp_idx = self.active_person.get_next_unset_keypoint_idx(0)
                    self.current_keypoint_idx_to_place = next_unset_kp_idx if next_unset_kp_idx is not None else 0
                status_text = (f"Edit P{self.active_person.id}. Current: {KEYPOINT_NAMES[self.current_keypoint_idx_to_place]}. "
                               f"LMB/RMB: Set Vis/Occ. MMB: Del KP. Q/E: Prev/Next KP.")
            elif self.active_person and self.active_person.is_suggestion_any_type():
                # Cannot directly edit a suggestion "shell"; must be accepted first. Revert to IDLE.
                self.current_mode = AnnotationMode.IDLE
                status_text = f"Selected P{self.active_person.id} (Suggestion Shell). Accept or Interpolate to edit."
            elif self.active_person and self.annotation_handler.is_person_done(self.active_person.id):
                # Cannot edit a "done" person. Revert to IDLE.
                self.current_mode = AnnotationMode.IDLE
                status_text = f"P{self.active_person.id} is marked Done (View only)."
            elif self.active_person and self._is_person_globally_hidden(self.active_person.id): 
                self.current_mode = AnnotationMode.IDLE # Revert to IDLE
                status_text = f"P{self.active_person.id} is Globally Hidden (View only)."
            else: # No valid active person for keypoint placement
                self.current_mode = AnnotationMode.IDLE
                status_text = "Error: No valid person selected for keypoint editing. Reverted to Idle."

        self.status_message_changed.emit(status_text)
        self._update_cursor_shape()
        self.mode_changed_by_canvas.emit(self.current_mode) # Notify main window
        self.update() # Request repaint

    def _update_cursor_shape(self):
        """Sets the mouse cursor shape based on the current mode and interaction."""
        is_active_person_done_and_real = (self.active_person and
                                          not self.active_person.is_suggestion_any_type() and
                                          self.annotation_handler.is_person_done(self.active_person.id))
        is_active_person_globally_hidden = self.active_person and self._is_person_globally_hidden(self.active_person.id)

        if self.dragged_bbox_corner_info is not None and not is_active_person_done_and_real and not is_active_person_globally_hidden:
            # Resizing bbox corners
            corner_type = self.dragged_bbox_corner_info[1]
            if corner_type == "TL" or corner_type == "BR": # TopLeft or BottomRight
                self.setCursor(Qt.CursorShape.SizeFDiagCursor) # Diagonal resize
            # Potentially add SizeVerCursor and SizeHorCursor if individual edge dragging is implemented
            else: # Fallback if other corner types were added
                self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif self.current_mode in [AnnotationMode.CREATING_BBOX_P1, AnnotationMode.CREATING_BBOX_P2]:
            self.setCursor(Qt.CursorShape.CrossCursor) # For placing bbox points
        elif self.current_mode == AnnotationMode.PLACING_KEYPOINTS and \
             self.active_person and not self.active_person.is_suggestion_any_type() and \
             not is_active_person_done_and_real and \
            not is_active_person_globally_hidden:
            self.setCursor(Qt.CursorShape.PointingHandCursor) # For placing/dragging keypoints
        elif self.is_panning:
            self.setCursor(Qt.CursorShape.SizeAllCursor) # For panning
        else: # Default
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # --- Coordinate Transformation Helpers ---
    def _widget_to_normalized_image_coords(self, widget_pos: QPointF | QPoint) -> QPointF:
        """Converts widget pixel coordinates to normalized image coordinates (0-1, 0-1 top-left)."""
        if self.width() == 0 or self.height() == 0: return QPointF(0, 0)

        pos = QPointF(widget_pos) if isinstance(widget_pos, QPoint) else widget_pos

        # 1. Widget (screen) to NDC (-1 to 1, -1 to 1 center, Y up)
        ndc_x = (pos.x() / self.width()) * 2.0 - 1.0
        ndc_y = 1.0 - (pos.y() / self.height()) * 2.0 # Y is inverted

        # 2. Apply inverse view transformation (pan, zoom) to get base image NDC
        # (ndc_coord - translation) / scale = base_ndc_coord
        base_ndc_x = (ndc_x - self.view_translation[0]) / self.view_scale
        base_ndc_y = (ndc_y - self.view_translation[1]) / self.view_scale

        # Clamp to image bounds in NDC (-1 to 1)
        base_ndc_x_clamped = max(-1.0, min(1.0, base_ndc_x))
        base_ndc_y_clamped = max(-1.0, min(1.0, base_ndc_y))

        # 3. Base image NDC to normalized image coordinates (0-1, 0-1 top-left)
        norm_img_x = (base_ndc_x_clamped + 1.0) / 2.0
        norm_img_y = (1.0 - base_ndc_y_clamped) / 2.0 # Y is inverted again for top-left origin

        return QPointF(norm_img_x, norm_img_y)

    def _normalized_image_to_render_ndc(self, normalized_image_pos: QPointF) -> QPointF:
        """Converts normalized image coordinates (0-1, 0-1 top-left) to renderable NDC for shaders."""
        # 1. Normalized image (0-1, 0-1 top-left) to base image NDC (-1 to 1, -1 to 1 center, Y up)
        base_ndc_x = normalized_image_pos.x() * 2.0 - 1.0
        base_ndc_y = 1.0 - normalized_image_pos.y() * 2.0 # Y is inverted

        # 2. Apply view transformation (pan, zoom)
        render_ndc_x = (base_ndc_x * self.view_scale) + self.view_translation[0]
        render_ndc_y = (base_ndc_y * self.view_scale) + self.view_translation[1]

        return QPointF(render_ndc_x, render_ndc_y)

    def _screen_pos_to_canvas_ndc(self, screen_pos: QPoint) -> QPointF:
        """Converts widget (screen) pixel coordinates directly to canvas NDC (-1 to 1, Y up for crosshairs)."""
        if self.width() == 0 or self.height() == 0: return QPointF(0,0)
        ndc_x = (screen_pos.x() / self.width()) * 2.0 - 1.0
        ndc_y = 1.0 - (screen_pos.y() / self.height()) * 2.0 # Y is inverted
        return QPointF(ndc_x, ndc_y)

    # --- Qt Event Handlers for User Interaction ---
    def keyPressEvent(self, event: QKeyEvent):
        """Handles key press events for mode switching, navigation, etc."""
        # Check if the active person is editable (real, not a suggestion, not "done")
        is_active_person_editable = (self.active_person and
                                        not self.active_person.is_suggestion_any_type() and
                                        not self.annotation_handler.is_person_done(self.active_person.id) and
                                        not self._is_person_globally_hidden(self.active_person.id))

        if self.current_mode == AnnotationMode.PLACING_KEYPOINTS and event.key() in [Qt.Key.Key_Return, Qt.Key.Key_Enter]:
            # Finish keypoint placement for the current person
            if is_active_person_editable:
                self.status_message_changed.emit(f"Person P{self.active_person.id} keypoints finalized.")
                self.dragged_keypoint_info = None # Clear any drag state
                self.set_annotation_mode(AnnotationMode.IDLE, self.active_person) # Switch to IDLE, keeping person active
                self.annotation_handler.save_annotations()
                self.persons_list_updated.emit(self.current_frame_display_index)
                self.annotation_action_completed.emit() # For auto-interpolation trigger
            event.accept()
            return

        elif event.key() == Qt.Key.Key_B: # Toggle BBox creation mode
            if self.current_mode == AnnotationMode.PLACING_KEYPOINTS:
                self.set_annotation_mode(AnnotationMode.IDLE, self.active_person) # Go to IDLE before BBox mode
            elif self.current_mode != AnnotationMode.CREATING_BBOX_P1:
                self.set_annotation_mode(AnnotationMode.CREATING_BBOX_P1)
            else: # Already in BBox mode, toggle to IDLE
                self.set_annotation_mode(AnnotationMode.IDLE)
            event.accept()
            return

        elif event.key() == Qt.Key.Key_Escape: # Cancel current action or deselect
            previous_active_person = self.active_person if self.current_mode == AnnotationMode.PLACING_KEYPOINTS and not self.dragged_keypoint_info else None
            self.dragged_keypoint_info = None # Cancel drag
            self.dragged_bbox_corner_info = None # Cancel bbox resize
            self.temp_bbox_start_norm_coords = None # Cancel bbox creation P2

            if self.current_mode == AnnotationMode.IDLE and self.active_person:
                previous_active_person = None # Deselect if already IDLE
            self.set_annotation_mode(AnnotationMode.IDLE, previous_active_person)
            event.accept()
            return

        elif event.key() == Qt.Key.Key_Q: # Previous keypoint in PLACING_KEYPOINTS mode
            if self.current_mode == AnnotationMode.PLACING_KEYPOINTS and is_active_person_editable and not self.dragged_keypoint_info:
                self.current_keypoint_idx_to_place = (self.current_keypoint_idx_to_place - 1 + NUM_KEYPOINTS) % NUM_KEYPOINTS
                self.status_message_changed.emit(f"Place: {KEYPOINT_NAMES[self.current_keypoint_idx_to_place]}")
                self.update()
            event.accept()
            return

        elif event.key() == Qt.Key.Key_E: # Next keypoint in PLACING_KEYPOINTS mode
            if self.current_mode == AnnotationMode.PLACING_KEYPOINTS and is_active_person_editable and not self.dragged_keypoint_info:
                self.current_keypoint_idx_to_place = (self.current_keypoint_idx_to_place + 1) % NUM_KEYPOINTS
                self.status_message_changed.emit(f"Place: {KEYPOINT_NAMES[self.current_keypoint_idx_to_place]}")
                self.update()
            event.accept()
            return

        super().keyPressEvent(event) # Pass to parent or Qt for default handling if not consumed

    def mousePressEvent(self, event: QMouseEvent):
        """Handles mouse button presses for various annotation actions."""
        if not self.hasFocus(): self.setFocus(Qt.FocusReason.MouseFocusReason) # Ensure widget has focus for key events

        norm_img_coords = self._widget_to_normalized_image_coords(event.position()) # Coords for annotation data

        # Check if the active person is real (not a suggestion) and editable (not "done")
        is_active_person_real_and_editable = (self.active_person and
                                                not self.active_person.is_suggestion_any_type() and
                                                not self.annotation_handler.is_person_done(self.active_person.id) and
                                                not self._is_person_globally_hidden(self.active_person.id))

        # --- Middle Mouse Button: Pan or Delete Keypoint ---
        if event.button() == Qt.MouseButton.MiddleButton:
            if self.current_mode == AnnotationMode.PLACING_KEYPOINTS and is_active_person_real_and_editable:
                # Try to delete an existing keypoint under the cursor
                clicked_kp_index = -1
                for i, (kp_x, kp_y, kp_vis) in enumerate(self.active_person.keypoints):
                    if kp_vis in [VISIBILITY_VISIBLE, VISIBILITY_OCCLUDED]: # Only delete set keypoints
                        kp_pos_norm = QPointF(kp_x, kp_y)
                        # Use a slightly larger hit radius for deletion for ease of use
                        hit_radius_for_delete = self.keypoint_drag_hit_radius_norm * 1.5
                        if (abs(norm_img_coords.x() - kp_pos_norm.x()) < hit_radius_for_delete and
                            abs(norm_img_coords.y() - kp_pos_norm.y()) < hit_radius_for_delete):
                            clicked_kp_index = i
                            break
                if clicked_kp_index != -1:
                    self.active_person.keypoints[clicked_kp_index] = [0.0, 0.0, VISIBILITY_NOT_SET] # Reset keypoint
                    kp_name = KEYPOINT_NAMES[clicked_kp_index]
                    self.status_message_changed.emit(f"Deleted '{kp_name}' for P{self.active_person.id}.")
                    self.current_keypoint_idx_to_place = clicked_kp_index # Set this as next to place
                    self.annotation_handler.save_annotations()
                    self.annotation_action_completed.emit()
                    self.persons_list_updated.emit(self.current_frame_display_index)
                    self.update()
                    event.accept()
                    return
            # If not deleting keypoint, start panning
            self.is_panning = True
            self.last_mouse_drag_pos = event.position()
            self._update_cursor_shape()
            event.accept()
            return

        # --- Left Mouse Button: Various Actions Based on Mode ---
        # Check for BBox corner drag initiation (if active person is editable and in IDLE/PLACING mode)
        if event.button() == Qt.MouseButton.LeftButton and \
           is_active_person_real_and_editable and \
           (self.current_mode == AnnotationMode.IDLE or self.current_mode == AnnotationMode.PLACING_KEYPOINTS) and \
           not self.dragged_keypoint_info: # Not already dragging a keypoint

            bbox_coords = self.active_person.bbox
            bbox_tl_norm = QPointF(bbox_coords[0], bbox_coords[1])
            bbox_br_norm = QPointF(bbox_coords[2], bbox_coords[3])
            bbox_tl_ndc = self._normalized_image_to_render_ndc(bbox_tl_norm)
            bbox_br_ndc = self._normalized_image_to_render_ndc(bbox_br_norm)
            mouse_render_ndc = self._screen_pos_to_canvas_ndc(event.position().toPoint()) # Mouse in render NDC

            # Check Top-Left corner hit
            if (abs(mouse_render_ndc.x() - bbox_tl_ndc.x()) < self.bbox_corner_draw_size_ndc and \
                abs(mouse_render_ndc.y() - bbox_tl_ndc.y()) < self.bbox_corner_draw_size_ndc):
                self.dragged_bbox_corner_info = (self.active_person, "TL")
                self.status_message_changed.emit(f"Adjusting Top-Left of BBox for P{self.active_person.id}")
                self._update_cursor_shape()
                self.update()
                event.accept()
                return
            # Check Bottom-Right corner hit
            elif (abs(mouse_render_ndc.x() - bbox_br_ndc.x()) < self.bbox_corner_draw_size_ndc and \
                  abs(mouse_render_ndc.y() - bbox_br_ndc.y()) < self.bbox_corner_draw_size_ndc):
                self.dragged_bbox_corner_info = (self.active_person, "BR")
                self.status_message_changed.emit(f"Adjusting Bottom-Right of BBox for P{self.active_person.id}")
                self._update_cursor_shape()
                self.update()
                event.accept()
                return

        # Click on a suggested keypoint to accept suggestion and start dragging
        # This occurs if PLACING_KEYPOINTS mode is active for a *real* person, but a suggestion for *that same person* is clicked.
        if self.current_mode == AnnotationMode.PLACING_KEYPOINTS and self.active_person and event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on a *suggested* keypoint of the *currently active real person*
            # This happens if a real person is active, but a suggestion for them (e.g. from interpolation) is visible and clicked.
            suggestions_on_frame = self.annotation_handler.get_suggested_annotations_for_frame(self.current_frame_display_index)
            clicked_suggestion_kp_info: tuple[PersonAnnotation.PersonAnnotation, int] | None = None # (suggested_person_obj, kp_idx)

            for sugg_person_obj in suggestions_on_frame:
                if sugg_person_obj.id == self.active_person.id: # Match ID with active real person
                    for kp_idx, (kp_x, kp_y, kp_vis) in enumerate(sugg_person_obj.keypoints):
                        if kp_vis in [VISIBILITY_SUGGESTED, VISIBILITY_AI_SUGGESTED]:
                            kp_pos_norm = QPointF(kp_x, kp_y)
                            hit_radius_for_sugg_accept = self.keypoint_drag_hit_radius_norm * 1.5
                            if (abs(norm_img_coords.x() - kp_pos_norm.x()) < hit_radius_for_sugg_accept and
                                abs(norm_img_coords.y() - kp_pos_norm.y()) < hit_radius_for_sugg_accept):
                                clicked_suggestion_kp_info = (sugg_person_obj, kp_idx)
                                break
                    if clicked_suggestion_kp_info: break

            if clicked_suggestion_kp_info:
                sugg_person_obj_clicked, sugg_kp_idx_clicked = clicked_suggestion_kp_info
                # Determine if auto-interpolation is on (affects how suggestions are cleared)
                is_auto_interpolate_active = False
                parent_window = self.parent() # Expected to be main Window
                if isinstance(parent_window, Window.Window):
                    is_auto_interpolate_active = parent_window.interpolation_panel.auto_interpolate_checkbox.isChecked()

                # Accept the entire suggestion for this person on this frame
                accepted_person = self.annotation_handler.accept_suggestion_for_frame(
                    self.current_frame_display_index, sugg_person_obj_clicked.id, is_auto_interpolate_active
                )
                if accepted_person:
                    self.active_person = accepted_person # Switch active person to the now real one
                    self.current_keypoint_idx_to_place = sugg_kp_idx_clicked # Set current kp
                    self.dragged_keypoint_info = (accepted_person, sugg_kp_idx_clicked) # Start dragging it
                    suggestion_type_str = "AI" if sugg_person_obj_clicked.has_ai_suggestions() else "Interpolated"
                    self.status_message_changed.emit(f"Accepted {suggestion_type_str} suggestion for P{accepted_person.id}. "
                                                     f"Now dragging '{KEYPOINT_NAMES[sugg_kp_idx_clicked]}'.")
                    self.persons_list_updated.emit(self.current_frame_display_index)
                    self.update()
                    event.accept()
                    return
                else:
                    self.status_message_changed.emit(f"Error: Could not accept suggestion for P{sugg_person_obj_clicked.id}.")
                    # Stay in current mode, don't accept event if accept failed

        # BBox Creation Logic
        if self.current_mode == AnnotationMode.CREATING_BBOX_P1 and event.button() == Qt.MouseButton.LeftButton:
            self.temp_bbox_start_norm_coords = norm_img_coords # Store first corner
            self.set_annotation_mode(AnnotationMode.CREATING_BBOX_P2) # Move to P2
            event.accept()
            return
        elif self.current_mode == AnnotationMode.CREATING_BBOX_P2 and event.button() == Qt.MouseButton.LeftButton:
            if not self.temp_bbox_start_norm_coords: # Should not happen if flow is correct
                self.set_annotation_mode(AnnotationMode.CREATING_BBOX_P1) # Reset
                event.accept()
                return

            p1_norm = self.temp_bbox_start_norm_coords
            p2_norm = norm_img_coords
            # Ensure x1 < x2 and y1 < y2
            x_min_norm = min(p1_norm.x(), p2_norm.x())
            y_min_norm = min(p1_norm.y(), p2_norm.y())
            x_max_norm = max(p1_norm.x(), p2_norm.x())
            y_max_norm = max(p1_norm.y(), p2_norm.y())

            # Prevent tiny bboxes
            if abs(x_min_norm - x_max_norm) < 0.005 or abs(y_min_norm - y_max_norm) < 0.005: # Threshold in normalized space
                self.status_message_changed.emit("Bounding box is too small. Please try again.")
                self.set_annotation_mode(AnnotationMode.CREATING_BBOX_P1) # Reset to P1
                event.accept()
                return

            new_person = PersonAnnotation.PersonAnnotation(person_id=-1, bbox=[x_min_norm, y_min_norm, x_max_norm, y_max_norm]) # -1 for new ID
            self.annotation_handler.add_person_to_frame(self.current_frame_display_index, new_person)
            self.annotation_handler.save_annotations()
            self.set_annotation_mode(AnnotationMode.PLACING_KEYPOINTS, new_person) # Switch to place keypoints for new person
            self.persons_list_updated.emit(self.current_frame_display_index)
            self.annotation_action_completed.emit()
            event.accept()
            return

        # Keypoint Placement/Selection/Drag Logic (if active person is editable)
        elif self.current_mode == AnnotationMode.PLACING_KEYPOINTS and is_active_person_real_and_editable:
            visibility_to_set = -1 # VISIBILITY_VISIBLE or VISIBILITY_OCCLUDED
            is_skeleton_complete = self.active_person.all_keypoints_set()
            clicked_existing_kp_index = -1

            # Check if clicking on an existing keypoint of the active person
            for i, (kp_x, kp_y, kp_vis) in enumerate(self.active_person.keypoints):
                if kp_vis in [VISIBILITY_VISIBLE, VISIBILITY_OCCLUDED]: # Only interact with set keypoints
                    kp_pos_norm = QPointF(kp_x, kp_y)
                    if (abs(norm_img_coords.x() - kp_pos_norm.x()) < self.keypoint_drag_hit_radius_norm and
                        abs(norm_img_coords.y() - kp_pos_norm.y()) < self.keypoint_drag_hit_radius_norm):
                        clicked_existing_kp_index = i
                        break

            if event.button() == Qt.MouseButton.LeftButton:
                if clicked_existing_kp_index != -1: # Clicked on an existing keypoint
                    if self.active_person.keypoints[clicked_existing_kp_index][2] != VISIBILITY_VISIBLE:
                        # If occluded, change to visible
                        self.active_person.keypoints[clicked_existing_kp_index][2] = VISIBILITY_VISIBLE
                        self.status_message_changed.emit(f"Set '{KEYPOINT_NAMES[clicked_existing_kp_index]}' to Visible for P{self.active_person.id}.")
                        self.annotation_handler.save_annotations(); self.annotation_action_completed.emit()
                        self.persons_list_updated.emit(self.current_frame_display_index); self.update()
                    else:
                        # If already visible, start dragging it
                        self.dragged_keypoint_info = (self.active_person, clicked_existing_kp_index)
                        self.current_keypoint_idx_to_place = clicked_existing_kp_index # Update current kp focus
                        self.status_message_changed.emit(f"Dragging '{KEYPOINT_NAMES[clicked_existing_kp_index]}' for P{self.active_person.id}.")
                        self.update()
                    event.accept(); return
                else: # Clicked on empty space
                    if is_skeleton_complete: event.accept(); return # Do nothing if all kps set
                    else: visibility_to_set = VISIBILITY_VISIBLE # Place new visible keypoint

            elif event.button() == Qt.MouseButton.RightButton:
                if self.dragged_keypoint_info: event.accept(); return # Don't change visibility if dragging
                if clicked_existing_kp_index != -1: # Right-clicked on an existing keypoint
                    if self.active_person.keypoints[clicked_existing_kp_index][2] != VISIBILITY_OCCLUDED:
                        # If visible, change to occluded
                        self.active_person.keypoints[clicked_existing_kp_index][2] = VISIBILITY_OCCLUDED
                        self.status_message_changed.emit(f"Set '{KEYPOINT_NAMES[clicked_existing_kp_index]}' to Occluded for P{self.active_person.id}.")
                        self.annotation_handler.save_annotations(); self.annotation_action_completed.emit()
                        self.persons_list_updated.emit(self.current_frame_display_index); self.update()
                    # If already occluded, do nothing on right-click
                    event.accept(); return
                else: # Right-clicked on empty space
                    if is_skeleton_complete: event.accept(); return # Do nothing
                    else: visibility_to_set = VISIBILITY_OCCLUDED # Place new occluded keypoint
            else: # Other mouse buttons
                if not event.isAccepted(): super().mousePressEvent(event)
                return

            # If placing a new keypoint (visibility_to_set is determined)
            if self.dragged_keypoint_info is None and visibility_to_set != -1:
                # Optional: Check if placing outside bbox (can be annoying, so commented out by default)
                person_bbox = self.active_person.bbox
                if not (person_bbox[0] == 0 and person_bbox[1] == 0 and person_bbox[2] == 0 and person_bbox[3] == 0): # If bbox is set
                    if not (person_bbox[0] <= norm_img_coords.x() <= person_bbox[2] and \
                            person_bbox[1] <= norm_img_coords.y() <= person_bbox[3]):
                        self.status_message_changed.emit("Keypoint must be placed inside the bounding box!")
                        event.accept(); return

                self.active_person.keypoints[self.current_keypoint_idx_to_place] = \
                    [norm_img_coords.x(), norm_img_coords.y(), visibility_to_set]

                # Advance to the next unset keypoint
                next_unset_idx = self.active_person.get_next_unset_keypoint_idx(self.current_keypoint_idx_to_place + 1)
                if next_unset_idx is None: # All keypoints are now set
                    if self.active_person.all_keypoints_set(): # Should be true here
                        self.status_message_changed.emit(f"P{self.active_person.id}: All keypoints set. Press Enter to finalize or Esc to continue editing.")
                        # self.set_annotation_mode(AnnotationMode.IDLE, self.active_person)
                        self.status_message_changed.emit(f"drag:")
                        self.annotation_handler.save_annotations()
                        self.annotation_action_completed.emit()
                    else: # Should not happen if get_next_unset_keypoint_idx is None
                        self.current_keypoint_idx_to_place = (self.current_keypoint_idx_to_place + 1) % NUM_KEYPOINTS
                        self.status_message_changed.emit(f"Place: {KEYPOINT_NAMES[self.current_keypoint_idx_to_place]}")
                else:
                    self.current_keypoint_idx_to_place = next_unset_idx
                    self.status_message_changed.emit(f"Place: {KEYPOINT_NAMES[next_unset_idx]}")

                self.persons_list_updated.emit(self.current_frame_display_index)
                self.update()
                event.accept()
                return

        # IDLE Mode: Select a person by clicking inside their BBox
        elif self.current_mode == AnnotationMode.IDLE and event.button() == Qt.MouseButton.LeftButton:
            # Iterate in reverse to select topmost person if bboxes overlap
            clicked_person = next(
                (p for p in reversed(self.annotation_handler.get_annotations_for_frame(self.current_frame_display_index))
                 if p.bbox != [0.0,0.0,0.0,0.0] and # Has a valid bbox
                    p.bbox[0] <= norm_img_coords.x() <= p.bbox[2] and # Click X within bbox X
                    p.bbox[1] <= norm_img_coords.y() <= p.bbox[3]), # Click Y within bbox Y
                None)

            if clicked_person:
                if self._is_person_globally_hidden(clicked_person.id):
                    # If hidden, select for viewing 
                    self.active_person = clicked_person
                    self.set_annotation_mode(AnnotationMode.IDLE, clicked_person) # Stays IDLE
                    self.status_message_changed.emit(f"Selected P{clicked_person.id} (Globally Hidden - View Only).")
                else: # Not hidden, select normally
                    self.active_person = clicked_person
                    self.set_annotation_mode(AnnotationMode.IDLE, clicked_person)
                event.accept()
                return

        if not event.isAccepted():
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handles mouse movement for panning, dragging keypoints/bboxes, and updating previews."""
        self.current_mouse_screen_pos = event.position().toPoint() # Update for text rendering etc.

        # Panning
        if self.is_panning and event.buttons() & Qt.MouseButton.MiddleButton:
            delta = event.position() - self.last_mouse_drag_pos
            self.last_mouse_drag_pos = event.position()
            # Convert screen delta to NDC delta (widget Y is inverted vs NDC Y)
            self.view_translation[0] += delta.x() * 2.0 / self.width()
            self.view_translation[1] += -delta.y() * 2.0 / self.height() # Negative for Y
            self.update()
            event.accept()
            return

        # Dragging BBox corner
        if self.dragged_bbox_corner_info and event.buttons() & Qt.MouseButton.LeftButton:
            person_modifying, corner_type = self.dragged_bbox_corner_info
            new_norm_coords = self._widget_to_normalized_image_coords(event.position())
            current_bbox = person_modifying.bbox

            if corner_type == "TL": # Modifying top-left
                # Prevent flipping by ensuring new TL is not past BR + small margin
                new_x1 = min(new_norm_coords.x(), current_bbox[2] - self.keypoint_drag_hit_radius_norm)
                new_y1 = min(new_norm_coords.y(), current_bbox[3] - self.keypoint_drag_hit_radius_norm)
                person_modifying.bbox[0] = max(0.0, new_x1) # Clamp to image bounds
                person_modifying.bbox[1] = max(0.0, new_y1)
            elif corner_type == "BR": # Modifying bottom-right
                new_x2 = max(new_norm_coords.x(), current_bbox[0] + self.keypoint_drag_hit_radius_norm)
                new_y2 = max(new_norm_coords.y(), current_bbox[1] + self.keypoint_drag_hit_radius_norm)
                person_modifying.bbox[2] = min(1.0, new_x2)
                person_modifying.bbox[3] = min(1.0, new_y2)
            # Note: After drag, mouseReleaseEvent will ensure bbox is [x_min, y_min, x_max, y_max]
            self.update()
            event.accept()
            return

        # Dragging Keypoint
        if self.dragged_keypoint_info and \
           (event.buttons() & Qt.MouseButton.LeftButton or event.buttons() & Qt.MouseButton.RightButton): # Allow drag with LMB or RMB
            person_modifying, kp_idx_dragging = self.dragged_keypoint_info
            new_norm_coords = self._widget_to_normalized_image_coords(event.position())

            # Clamp keypoint position to within the person's bounding box
            person_bbox = person_modifying.bbox
            clamped_x = max(person_bbox[0], min(new_norm_coords.x(), person_bbox[2]))
            clamped_y = max(person_bbox[1], min(new_norm_coords.y(), person_bbox[3]))

            person_modifying.keypoints[kp_idx_dragging][0] = clamped_x
            person_modifying.keypoints[kp_idx_dragging][1] = clamped_y
            self.update()
            event.accept()
            return

        # Update display if in a mode that shows previews (e.g., crosshairs, temp bbox)
        if self.current_mode in [AnnotationMode.CREATING_BBOX_P1,
                                 AnnotationMode.CREATING_BBOX_P2,
                                 AnnotationMode.PLACING_KEYPOINTS]: # For text near cursor
            self.update()
            event.accept()
            return

        if not event.isAccepted():
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handles mouse button releases to finalize actions like panning or dragging."""
        # Stop Panning
        if event.button() == Qt.MouseButton.MiddleButton and self.is_panning:
            self.is_panning = False
            self._update_cursor_shape()
            event.accept()
            return

        # Finalize BBox corner drag
        if event.button() == Qt.MouseButton.LeftButton and self.dragged_bbox_corner_info:
            person_modified, corner_dragged = self.dragged_bbox_corner_info
            # Ensure bbox coords are ordered (x1 < x2, y1 < y2)
            x1, y1, x2, y2 = person_modified.bbox[0], person_modified.bbox[1], person_modified.bbox[2], person_modified.bbox[3]
            person_modified.bbox = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]

            self.status_message_changed.emit(f"BBox for P{person_modified.id} adjusted.")
            self.dragged_bbox_corner_info = None # Clear drag state
            self.annotation_handler.save_annotations()
            self.annotation_action_completed.emit()
            self._update_cursor_shape()
            self.update()
            event.accept()
            return

        # Finalize Keypoint drag
        if (event.button() == Qt.MouseButton.LeftButton and self.dragged_keypoint_info) or \
           (event.button() == Qt.MouseButton.RightButton and self.dragged_keypoint_info): # Drag can be with LMB or RMB
            person_modified, kp_idx_dragged = self.dragged_keypoint_info
            self.status_message_changed.emit(f"Finished dragging '{KEYPOINT_NAMES[kp_idx_dragged]}' for P{person_modified.id}.")
            self.dragged_keypoint_info = None # Clear drag state
            self.annotation_handler.save_annotations()
            self.persons_list_updated.emit(self.current_frame_display_index)
            # Return to PLACING_KEYPOINTS mode for this person, ready for next action
            self.set_annotation_mode(AnnotationMode.PLACING_KEYPOINTS, person_modified)
            self.annotation_action_completed.emit()
            self.update()
            event.accept()
            return

        if not event.isAccepted():
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handles mouse double-clicks, e.g., to quickly edit a person or accept suggestions."""
        if not self.hasFocus(): self.setFocus(Qt.FocusReason.MouseFocusReason)

        if event.button() == Qt.MouseButton.LeftButton:
            norm_img_coords = self._widget_to_normalized_image_coords(event.position())

            # 1. Try to accept a suggestion if double-clicked on one
            suggestion_to_accept_on_dclick: PersonAnnotation.PersonAnnotation | None = None
            for sugg_p in self.annotation_handler.get_suggested_annotations_for_frame(self.current_frame_display_index):
                for i, (kp_x, kp_y, kp_vis) in enumerate(sugg_p.keypoints):
                    if kp_vis in [VISIBILITY_SUGGESTED, VISIBILITY_AI_SUGGESTED]:
                        kp_pos_norm = QPointF(kp_x, kp_y)
                        # Use a larger hit radius for double-click accept for convenience
                        hit_radius_for_dclick_accept = self.keypoint_drag_hit_radius_norm * 2.5
                        if (abs(norm_img_coords.x() - kp_pos_norm.x()) < hit_radius_for_dclick_accept and
                            abs(norm_img_coords.y() - kp_pos_norm.y()) < hit_radius_for_dclick_accept):
                            suggestion_to_accept_on_dclick = sugg_p
                            break
                if suggestion_to_accept_on_dclick: break

            if suggestion_to_accept_on_dclick:
                is_auto_interpolate_active = False
                parent_window = self.parent()
                if isinstance(parent_window, Window.Window):
                    is_auto_interpolate_active = parent_window.interpolation_panel.auto_interpolate_checkbox.isChecked()

                accepted_real_person = self.annotation_handler.accept_suggestion_for_frame(
                    self.current_frame_display_index, suggestion_to_accept_on_dclick.id, is_auto_interpolate_active
                )
                if accepted_real_person:
                    self.active_person = accepted_real_person
                    self.current_keypoint_idx_to_place = 0 # Default to first keypoint for editing
                    self.set_annotation_mode(AnnotationMode.PLACING_KEYPOINTS, accepted_real_person)
                    sugg_type_str = "AI" if suggestion_to_accept_on_dclick.has_ai_suggestions() else "Interpolated"
                    self.status_message_changed.emit(f"{sugg_type_str} suggestion for P{accepted_real_person.id} accepted by double-click. Now editing.")
                    self.persons_list_updated.emit(self.current_frame_display_index)
                    self.update()
                    event.accept()
                    return
                else:
                    self.status_message_changed.emit(f"Failed to accept suggestion for P{suggestion_to_accept_on_dclick.id} on double-click.")
                    # Don't accept event if fail, allow fallback to person edit

            # 2. If not a suggestion, try to activate editing for a real, non-done person by double-clicking their keypoint
            person_to_edit_on_dclick: PersonAnnotation.PersonAnnotation | None = None
            keypoint_idx_dclicked: int = -1
            # Iterate in reverse to pick top-most person if overlapping
            real_persons_on_frame = reversed(self.annotation_handler.get_annotations_for_frame(self.current_frame_display_index))
            for p_real in real_persons_on_frame:
                if p_real.is_suggestion_any_type(): continue # Should not happen in real_persons list but safeguard
                for i, (kp_x, kp_y, kp_vis) in enumerate(p_real.keypoints):
                    if kp_vis in [VISIBILITY_VISIBLE, VISIBILITY_OCCLUDED]: # Only on set keypoints
                        kp_pos_norm = QPointF(kp_x, kp_y)
                        hit_radius_for_dclick_edit = self.keypoint_drag_hit_radius_norm * 2.0
                        if (abs(norm_img_coords.x() - kp_pos_norm.x()) < hit_radius_for_dclick_edit and
                            abs(norm_img_coords.y() - kp_pos_norm.y()) < hit_radius_for_dclick_edit):
                            person_to_edit_on_dclick = p_real
                            keypoint_idx_dclicked = i
                            break
                if person_to_edit_on_dclick: break

            if person_to_edit_on_dclick is not None and keypoint_idx_dclicked != -1 and \
               not self.annotation_handler.is_person_done(person_to_edit_on_dclick.id) and \
                not self._is_person_globally_hidden(person_to_edit_on_dclick.id):
                self.active_person = person_to_edit_on_dclick
                self.current_keypoint_idx_to_place = keypoint_idx_dclicked
                self.set_annotation_mode(AnnotationMode.PLACING_KEYPOINTS, self.active_person)
                self.status_message_changed.emit(f"Editing P{self.active_person.id} by double-click. Current: {KEYPOINT_NAMES[keypoint_idx_dclicked]}.")
                self.persons_list_updated.emit(self.current_frame_display_index)
                self.update()
                event.accept()
                return

        if not event.isAccepted():
            super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Handles mouse wheel events for zooming the canvas."""
        angle_delta = event.angleDelta().y() / 8.0  # Degrees
        normalized_delta = angle_delta / 15.0    # Standard mouse wheel step

        zoom_factor_step = 1.10
        zoom_factor = pow(zoom_factor_step, normalized_delta)

        # Zoom centered on mouse cursor
        mouse_pos_widget = event.position()
        # Convert mouse widget pos to NDC (-1 to 1, Y up) for zoom calculation
        mouse_ndc_x = (mouse_pos_widget.x() / self.width()) * 2.0 - 1.0
        mouse_ndc_y = 1.0 - (mouse_pos_widget.y() / self.height()) * 2.0

        old_scale = self.view_scale
        self.view_scale = max(0.01, min(self.view_scale * zoom_factor, 100.0)) # Clamp zoom

        # Adjust translation to keep point under mouse stationary
        # P_ndc_mouse = (P_base_ndc * old_scale) + old_translation
        # P_ndc_mouse = (P_base_ndc * new_scale) + new_translation
        # Solve for new_translation:
        # new_translation_x = P_ndc_mouse_x * (1 - new_scale/old_scale) + old_translation_x * (new_scale/old_scale)
        self.view_translation[0] = mouse_ndc_x * (1.0 - self.view_scale / old_scale) + \
                                   self.view_translation[0] * (self.view_scale / old_scale)
        self.view_translation[1] = mouse_ndc_y * (1.0 - self.view_scale / old_scale) + \
                                   self.view_translation[1] * (self.view_scale / old_scale)
        self.update()
        event.accept()

    def _is_person_globally_hidden(self, person_id: int) -> bool:
        """Checks if a person ID is in the main window's globally_hidden_person_ids set."""
        top_level_window = self.window() # Get the QMainWindow instance
        if isinstance(top_level_window, Window.Window): 
            return person_id in top_level_window.globally_hidden_person_ids
        return False 
    
    # --- Public Methods for External Control ---
    def activate_person_for_editing(self, person_id: int, is_suggestion: bool = False) -> bool:
        """
        Activates a person for editing, potentially accepting a suggestion first.
        Args:
            person_id: The ID of the person to activate.
            is_suggestion: True if the person_id refers to a suggested annotation.
        Returns:
            True if activation was successful, False otherwise.
        """
        if is_suggestion:
            suggestion_obj = next((p for p in self.annotation_handler.get_suggested_annotations_for_frame(self.current_frame_display_index)
                                   if p.id == person_id), None)
            if suggestion_obj:
                is_auto_interpolate_active = False # Default, get from main window if possible
                parent_window = self.parent()
                if isinstance(parent_window, Window.Window):
                    is_auto_interpolate_active = parent_window.interpolation_panel.auto_interpolate_checkbox.isChecked()

                accepted_person = self.annotation_handler.accept_suggestion_for_frame(
                    self.current_frame_display_index, person_id, is_auto_interpolate_active
                )
                if accepted_person:
                    self.active_person = accepted_person
                    self.set_annotation_mode(AnnotationMode.PLACING_KEYPOINTS, accepted_person)
                    self.status_message_changed.emit(f"Suggestion for P{person_id} accepted. Now editing.")
                    self.persons_list_updated.emit(self.current_frame_display_index)
                    return True
                else:
                    self.status_message_changed.emit(f"Could not accept suggestion for P{person_id}.")
                    return False
            else:
                self.status_message_changed.emit(f"Suggested P{person_id} not found on current frame.")
                return False
        else: # Activating a real annotation
            real_person_obj = self.annotation_handler.get_person_by_id_in_frame(self.current_frame_display_index, person_id)
            if real_person_obj:
                if self.annotation_handler.is_person_done(real_person_obj.id):
                    self.status_message_changed.emit(f"P{real_person_obj.id} is marked Done. Cannot edit. Select to view.")
                    self.set_annotation_mode(AnnotationMode.IDLE, real_person_obj) # Allow selection for viewing
                    return False # Not editable, but selected
                elif self._is_person_globally_hidden(real_person_obj.id): 
                    self.status_message_changed.emit(f"P{real_person_obj.id} is Globally Hidden. Cannot edit. Select to view.")
                    self.set_annotation_mode(AnnotationMode.IDLE, real_person_obj)
                    return False
                else:
                    self.set_annotation_mode(AnnotationMode.PLACING_KEYPOINTS, real_person_obj)
                    return True
            else:
                self.status_message_changed.emit(f"Real P{person_id} not found on current frame.")
                return False
        return False # Should not be reached

    def clear_all_annotations_on_current_frame(self):
        """Clears all real and suggested annotations from the currently displayed frame."""
        current_frame_idx = self.current_frame_display_index
        # Clear real annotations
        if current_frame_idx in self.annotation_handler.all_annotations_by_frame:
            self.annotation_handler.all_annotations_by_frame[current_frame_idx] = []
            # Optionally, remove the frame entry if it's empty, or keep it as an empty list
            # del self.annotation_handler.all_annotations_by_frame[current_frame_idx]
            self.annotation_handler.save_annotations() # Persist removal of real annotations

        # Clear suggested annotations
        if current_frame_idx in self.annotation_handler.suggested_annotations_by_frame:
            del self.annotation_handler.suggested_annotations_by_frame[current_frame_idx]
            # No save needed for suggestions as they are transient or re-generated.

        self.active_person = None
        self.set_annotation_mode(AnnotationMode.IDLE)
        self.persons_list_updated.emit(current_frame_idx)
        self.status_message_changed.emit(f"All annotations and suggestions cleared for Frame {current_frame_idx + 1}.")
        self.update()