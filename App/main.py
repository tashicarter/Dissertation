import tempfile
import numpy as np

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from ultralytics import YOLO
from detection import create_colors_info, detect

def main():

    st.set_page_config(page_title="Dissertation", layout="wide", initial_sidebar_state="expanded")
    st.title("Basketball Players Detection With Team Prediction & Tactical Map", anchor=False)
    st.header("Works only with Tactical Camera footage", anchor=False)

    st.sidebar.title("Main Settings")
    demo_selected = st.sidebar.radio(label="Select Demo Video", options=["Demo 1", "Demo 2"], horizontal=True)

    ## Sidebar Setup
    st.sidebar.markdown('---')
    st.sidebar.header("Video Upload")
    input_vide_file = st.sidebar.file_uploader(label='Upload a video file', type=['mp4','mov', 'avi', 'm4v', 'asf'])

    demo_vid_paths={
        "Demo 1":'./demo_vid_1.mp4',
        "Demo 2":'./demo_vid_2.mp4'
    }
    demo_vid_path = demo_vid_paths[demo_selected]
    demo_team_info = {
        "Demo 1":{"team1_name":"Golden State Warriors",
                  "team2_name":"Detroit Pistons",
                  "team1_p_color":'#1E2530',
                  "team2_p_color":'#FBFCFA',
                  },
        "Demo 2":{"team1_name":"Philadephia 76ers",
                  "team2_name":"Phoenix Suns",
                  "team1_p_color":'#29478A',
                  "team2_p_color":'#90C8FF',
                  }
    }
    selected_team_info = demo_team_info[demo_selected]

    tempf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    if not input_vide_file:
        tempf.name = demo_vid_path
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()

        st.sidebar.text('Demo video')
        st.sidebar.video(demo_bytes)
    else:
        tempf.write(input_vide_file.read())
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()

        st.sidebar.text('Input video')
        st.sidebar.video(demo_bytes)


    st.sidebar.markdown('---')
    st.sidebar.subheader("Team Names")
    team1_name = st.sidebar.text_input(label='First Team Name', value=selected_team_info["team1_name"])
    team2_name = st.sidebar.text_input(label='Second Team Name', value=selected_team_info["team2_name"])
    st.sidebar.markdown('---')

    ## Page Setup
    tab1, tab2, tab3, tab4 = st.tabs(["Instructions", "Model Selection", "Team Colors", "Model Hyperparameters & Detection"])
    with tab1:
        st.header(':blue[Welcome!]', anchor=False)
        st.subheader('Main Application Functionalities:', divider='blue', anchor=False)
        st.markdown("""
                    1. Basketball players, referee, and ball detection.
                    2. Players team prediction.
                    3. Estimation of players positions on a tactical map.
                    """)
        st.subheader('How to use the application', anchor=False, divider='blue')
        st.markdown("""
                    **There are two demo videos that are automaticaly loaded when you start the app, alongside the recommended settings and hyperparameters**
                    1. Upload a video to analyse, using the sidebar menu "Browse files" button.
                    2. Enter the team names that corresponds to the uploaded video in the text fields in the sidebar menu.
                    3. Access the "Team colors" tab in the main page.
                    4. Select a frame where players from both teams can be detected.
                    5. Follow the instruction on the page to pick each team colors.
                    6. Go to the "Model Hyperparameters & Detection" tab, adjust hyperparameters and select the annotation options. (Default hyperparameters are recommended)
                    7. Run Detection!
                    8. If "save outputs" option was selected the saved video can be found in the "outputs" directory
                    """)

    with tab2:
        st.header("YOLOv8L is highly recommended!", anchor=False)
        st.subheader("Smaller sizes will speed up the detection process, however, will lose accuracy and may potentially cause the application to crash if the minimum number of court key points (4) haven't been detected.", anchor=False, divider='blue')
        st.subheader("Player Model Selection", anchor=False)
        selected_model = st.radio("Select YOLO Model", ("YOLOv8s", "YOLOv8m", "YOLOv8L"), index=2, horizontal=True, key="player_model")

        if selected_model == "YOLOv8s":
            # Code to load YOLOv8s model
            st.write("You selected YOLOv8s")
            # Load the YOLOv8 players detection model
            model_players = YOLO("../models/Yolov8S Players/weights/best.pt")
        elif selected_model == "YOLOv8m":
            # Code to load YOLOv8m model
            st.write("You selected YOLOv8m")
            # Load the YOLOv8 players detection model
            model_players = YOLO("../models/Yolov8M Players/weights/best.pt")
        elif selected_model == "YOLOv8L":
            st.write("You selected YOLOv8L")
            # Load the YOLOv8 players detection model
            model_players = YOLO("../models/Yolov8L Players/weights/best.pt")

        st.subheader("Keypoint Model Selection", anchor=False)
        selected_modelKP = st.radio("Select YOLO Model", ("YOLOv8s", "YOLOv8m", "YOLOv8L"), index=2, horizontal=True, key="keypoint_model")

        if selected_modelKP == "YOLOv8s":
            # Code to load YOLOv8s model
            st.write("You selected YOLOv8s")
            # Load the YOLOv8 KP detection model
            model_keypoints = YOLO("../models/Yolov8SKP/weights/best.pt")
        elif selected_modelKP == "YOLOv8m":
            # Code to load YOLOv8m model
            st.write("You selected YOLOv8m")
            # Load the YOLOv8 KP detection model
            model_keypoints = YOLO("../models/Yolov8MKP/weights/best.pt")
        elif selected_modelKP == "YOLOv8L":
            st.write("You selected YOLOv8L")
            # Load the YOLOv8 KP detection model
            model_keypoints = YOLO("../models/Yolov8LKP/weights/best.pt")


    with tab3:
        t1col1, t1col2 = st.columns([1,1])
        with t1col1:
            cap_temp = cv2.VideoCapture(tempf.name)
            frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_nbr = st.slider(label="Select frame", min_value=1, max_value=frame_count, step=1, help="Select frame to pick team colors from")
            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
            success, frame = cap_temp.read()
            with st.spinner('Detecting players in selected frame..'):
                results = model_players(frame, conf=0.7)
                bboxes = results[0].boxes.xyxy.cpu().numpy()
                labels = results[0].boxes.cls.cpu().numpy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections_imgs_list = []
                detections_imgs_grid = []
                padding_img = np.ones((80,60,3),dtype=np.uint8)*255
                for i, j in enumerate(list(labels)):
                    if int(j) == 1:
                        bbox = bboxes[i,:]                         
                        obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                        obj_img = cv2.resize(obj_img, (60,80))
                        detections_imgs_list.append(obj_img)
                detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list)//2)])
                detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list)//2, len(detections_imgs_list))])
                if len(detections_imgs_list)%2 != 0:
                    detections_imgs_grid[0].append(padding_img)
                concat_det_imgs_row1 = cv2.hconcat(detections_imgs_grid[0])
                concat_det_imgs_row2 = cv2.hconcat(detections_imgs_grid[1])
                concat_det_imgs = cv2.vconcat([concat_det_imgs_row1,concat_det_imgs_row2])
            st.write("Detected players")
            value = streamlit_image_coordinates(concat_det_imgs, key="numpy")
            #value_radio_dic = defaultdict(lambda: None)
            st.markdown('---')
            radio_options =[f"{team1_name} P color", f"{team2_name} P color"]
            active_color = st.radio(label="Select which team color to pick from the image above", options=radio_options, horizontal=True,
                                    help="Choose team color you want to pick and click on the image above to pick the color. Colors will be displayed in boxes below.")
            if value is not None:
                picked_color = concat_det_imgs[value['y'], value['x'], :]
                st.session_state[f"{active_color}"] = '#%02x%02x%02x' % tuple(picked_color)
            st.write("Boxes below can be used to manually adjust selected colors.")
            cp1, cp2 = st.columns([1,1])
            with cp1:
                hex_color_1 = st.session_state[f"{team1_name} P color"] if f"{team1_name} P color" in st.session_state else selected_team_info["team1_p_color"]
                team1_p_color = st.color_picker(label='Team 1 colour', value=hex_color_1, key='t1p')
                st.session_state[f"{team1_name} P color"] = team1_p_color
            with cp2:
                hex_color_3 = st.session_state[f"{team2_name} P color"] if f"{team2_name} P color" in st.session_state else selected_team_info["team2_p_color"]
                team2_p_color = st.color_picker(label='Team 2 colour', value=hex_color_3, key='t2p')
                st.session_state[f"{team2_name} P color"] = team2_p_color
        st.markdown('---')

        with t1col2:
            extracted_frame = st.empty()
            extracted_frame.image(frame, use_column_width=True, channels="BGR")

        
    colors_dic, color_list_lab = create_colors_info(team1_name, st.session_state[f"{team1_name} P color"], team2_name, st.session_state[f"{team2_name} P color"])

    with tab4:
        t2col1, t2col2 = st.columns([1,1])
        with t2col1:
            player_model_conf_thresh = st.slider('Players Detection Confidence Threshold', min_value=0.0, max_value=1.0, value=0.7)
            keypoints_model_conf_thresh = st.slider('Field Keypoints Players Detection Confidence Threshold', min_value=0.0, max_value=1.0, value=0.15)
            keypoints_displacement_mean_tol = st.slider('Keypoints Displacement RMSE Tolerance (pixels)', min_value=-1, max_value=100, value=7,
                                                         help="Indicates the maximum allowed average distance between the position of the field keypoints\
                                                           in current and previous detections. It is used to determine wether to update homography matrix or not. ")
            detection_hyper_params = {
                0: player_model_conf_thresh,
                1: keypoints_model_conf_thresh,
                2: keypoints_displacement_mean_tol
            }
        with t2col2:
            num_pal_colors = st.slider(label="Number of palette colors", min_value=1, max_value=5, step=1, value=3,
                                    help="How many colors to extract form detected players bounding-boxes? It is used for team prediction.")
            st.markdown("---")
            save_output = st.checkbox(label='Save output', value=False)
            if save_output:
                output_file_name = st.text_input(label='File Name (Optional)', placeholder='Enter output video file name.')
            else:
                output_file_name = None
        st.markdown("---")

        
        bcol1, bcol2 = st.columns([1,1])
        with bcol2:
            st.write("Annotation options:")
            bcol21t, bcol22t = st.columns([1,1])
            with bcol21t:
                show_k = st.toggle(label="Show Keypoints Detections", value=False)
                show_p = st.toggle(label="Show Players Detections", value=True)
            with bcol22t:
                show_pal = st.toggle(label="Show Color Palettes", value=True)
            plot_hyperparams = {
                0: show_k,
                1: show_pal,
                2: show_p
            }
            st.markdown('---')
            bcol21, bcol22, bcol23, bcol24 = st.columns([1.5,1,1,1])
            with bcol21:
                st.write('')
            with bcol22:
                ready = True if (team1_name == '') or (team2_name == '') else False
                start_detection = st.button(label='Start Detection', disabled=ready)
            with bcol23:
                stop_btn_state = True if not start_detection else False
                stop_detection = st.button(label='Stop Detection', disabled=stop_btn_state)
            with bcol24:
                st.write('')


    stframe = st.empty()
    cap = cv2.VideoCapture(tempf.name)
    status = False

    if start_detection and not stop_detection:
        st.toast(f'Detection Started!')
        status = detect(cap, stframe, output_file_name, save_output, model_players, model_keypoints,
                         detection_hyper_params, plot_hyperparams,
                           num_pal_colors, colors_dic, color_list_lab)
    else:
        try:
            # Release the video capture object and close the display window
            cap.release()
        except:
            pass
    if status:
        st.toast(f'Detection Completed!')
        cap.release()


if __name__=='__main__':
    try:
        main()
    except SystemExit:
        pass