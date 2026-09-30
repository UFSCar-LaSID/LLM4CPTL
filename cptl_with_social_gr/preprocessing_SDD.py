import os
import glob
from collections import defaultdict

# ==========================================
# THRESHOLDS DE CURADORIA DO DATASET
# ==========================================
MIN_PEDESTRIANS_PER_VIDEO = 100  
MAX_VIDEOS_PER_SCENE = 1  
MIN_TRAJ_LENGTH = 20            

SDD_SCALES = {
    "bookstore": {"video0": 0.038392063, "video1": 0.039892913, "video2": 0.04062433, "video3": 0.039098596, "video4": 0.0396, "video5": 0.0396, "video6": 0.0413},
    "coupa": {"video0": 0.027995674, "video1": 0.023224545, "video2": 0.024, "video3": 0.025524906},
    "deathCircle": {"video0": 0.04064, "video1": 0.039076923, "video2": 0.03948382, "video3": 0.028478209, "video4": 0.038980137},
    "gates": {"video0": 0.03976968, "video1": 0.03770837, "video2": 0.037272793, "video3": 0.034515323, "video4": 0.04412268, "video5": 0.0342392, "video6": 0.0342392, "video7": 0.04540353, "video8": 0.045191525},
    "hyang": {"video0": 0.034749693, "video1": 0.0453136, "video2": 0.054992233, "video3": 0.056642, "video4": 0.034265612, "video5": 0.029655497, "video6": 0.052936449, "video7": 0.03540125, "video8": 0.034592381, "video9": 0.038031423, "video10": 0.054460944, "video11": 0.054992233, "video12": 0.054104065, "video13": 0.0541, "video14": 0.0541},
    "little": {"video0": 0.028930169, "video1": 0.028543144, "video2": 0.028543144, "video3": 0.028638926},
    "nexus": {"video0": 0.043986494, "video1": 0.043316805, "video2": 0.042247434, "video3": 0.045883871, "video4": 0.045883871, "video5": 0.045395745, "video6": 0.037929168, "video7": 0.037106087, "video8": 0.037106087, "video9": 0.044917895, "video10": 0.043991753, "video11": 0.043766154},
    "quad": {"video0": 0.043606807, "video1": 0.042530206, "video2": 0.043338169, "video3": 0.044396842}
}

def preprocess_sdd(raw_dir, output_dir):
    annotation_files = glob.glob(os.path.join(raw_dir, 'annotations', '*', '*', 'annotations.txt'))
    
    if not annotation_files:
        print("Nenhum arquivo de anotação encontrado.")
        return

    print("ETAPA 1: Extração e segmentação espacial contínua...")
    processed_data = defaultdict(dict)

    for filepath in annotation_files:
        parts = filepath.split(os.sep)
        scene_name = parts[-3]
        video_name = parts[-2]
        
        scale_factor = SDD_SCALES.get(scene_name, {}).get(video_name, 1.0)
        trajectories_by_track = defaultdict(list)

        with open(filepath, 'r') as file:
            for line in file:
                data = line.strip().split()
                if len(data) < 10:
                    continue
                
                track_id = data[0]
                xmin, ymin = float(data[1]), float(data[2])
                xmax, ymax = float(data[3]), float(data[4])
                frame = float(data[5])
                lost, occluded = int(data[6]), int(data[7])
                label = data[9]

                if int(frame) % 12 != 0:
                    continue

                is_pedestrian = (label == '"Pedestrian"' or label == 'Pedestrian')
                is_valid = (lost == 0 and occluded == 0)

                if is_pedestrian and is_valid:
                    x_metros = ((xmin + xmax) / 2.0) * scale_factor
                    y_metros = ((ymin + ymax) / 2.0) * scale_factor
                    trajectories_by_track[track_id].append((frame, x_metros, y_metros))

        if not trajectories_by_track:
            continue

        all_frames = sorted(list(set(frame for track in trajectories_by_track.values() for frame, _, _ in track)))
        expected_step = min([all_frames[i] - all_frames[i-1] for i in range(1, len(all_frames))]) if len(all_frames) > 1 else 1.0

        first_frame = all_frames[0] if all_frames else 0.0

        final_video_trajectories = []
        new_global_track_id = 1.0
        total_pedestrians_in_video = 0

        for orig_track_id, points in trajectories_by_track.items():
            points.sort(key=lambda item: item[0])
            if len(points) < MIN_TRAJ_LENGTH:
                continue
                
            current_segment = [points[0]]
            for i in range(1, len(points)):
                curr_frame = points[i][0]
                prev_frame = points[i-1][0]
                
                if curr_frame - prev_frame > expected_step:
                    if len(current_segment) >= MIN_TRAJ_LENGTH:
                        for pt in current_segment:
                            mapped_frame = first_frame + ((pt[0] - first_frame) / 12.0)
                            final_video_trajectories.append((mapped_frame, new_global_track_id, pt[1], pt[2]))
                        new_global_track_id += 1.0
                        total_pedestrians_in_video += 1
                    current_segment = [points[i]]
                else:
                    current_segment.append(points[i])
                    
            if len(current_segment) >= MIN_TRAJ_LENGTH:
                for pt in current_segment:
                    mapped_frame = first_frame + ((pt[0] - first_frame) / 12.0)
                    final_video_trajectories.append((mapped_frame, new_global_track_id, pt[1], pt[2]))
                new_global_track_id += 1.0
                total_pedestrians_in_video += 1

        if final_video_trajectories:
            processed_data[scene_name][video_name] = {
                'count': total_pedestrians_in_video,
                'trajectories': final_video_trajectories
            }

    print("\nETAPA 2: Curadoria de Dados (Filtro de Densidade e Top-K)...")
    surviving_videos = []
    
    for scene, videos in processed_data.items():
        # Avalia quais vídeos desta cena superam o limiar estrito
        valid_videos = {v: d for v, d in videos.items() if d['count'] >= MIN_PEDESTRIANS_PER_VIDEO}
        
        # O mecanismo de Fallback: 
        # Se nenhum vídeo bateu a meta, escolhemos forçadamente o melhor daquela cena
        if not valid_videos:
            best_video_name = max(videos, key=lambda v: videos[v]['count'])
            top_videos = [(best_video_name, videos[best_video_name])]
            print(f"  [Fallback Resgate] O cenário '{scene}' reprovou no threshold. Resgatando o melhor vídeo ({best_video_name}).")
        else:
            sorted_videos = sorted(valid_videos.items(), key=lambda x: x[1]['count'], reverse=True)
            top_videos = sorted_videos[:MAX_VIDEOS_PER_SCENE]
        
        for v_name, v_data in top_videos:
            surviving_videos.append({
                'scene': scene,
                'video': v_name,
                'trajs': v_data['trajectories']
            })
            print(f"  [Mantido] {scene}_{v_name} | Pedestres: {v_data['count']}")

    print(f"\nETAPA 3: Fatiamento Temporal balanceado por volume para os {len(surviving_videos)} vídeos...")
    
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    for video_dict in surviving_videos:
        scene = video_dict['scene']
        video = video_dict['video']
        trajs = video_dict['trajs'] 
        
        # Agrupa os pontos por track_id
        trajs_by_track = defaultdict(list)
        for t in trajs:
            trajs_by_track[t[1]].append(t)
            
        # 1. Cria uma lista de "agentes", onde cada agente é [track_id, start_frame, points]
        agents = []
        for track_id, points in trajs_by_track.items():
            start_frame = points[0][0]
            agents.append((track_id, start_frame, points))
            
        # 2. Ordena os agentes no tempo, baseando-se em QUANDO eles aparecem no vídeo
        agents.sort(key=lambda x: x[1])
        
        total_agents = len(agents)
        train_thresh = int(total_agents * 0.7)
        val_thresh = int(total_agents * 0.8)
        
        train_trajs, val_trajs, test_trajs = [], [], []
        
        # 3. Distribui exatamente 70% das PESSOAS para o treino, 10% val, 20% teste.
        for idx, (track_id, start_frame, points) in enumerate(agents):
            if idx < train_thresh:
                train_trajs.extend(points)
            elif idx < val_thresh:
                val_trajs.extend(points)
            else:
                test_trajs.extend(points)
                
        # Escreve nos respectivos arquivos dentro das três pastas
        for split_name, split_data in [('train', train_trajs), ('val', val_trajs), ('test', test_trajs)]:
            if split_data:
                # Ordena os blocos no tempo para o modelo ler em ordem cronológica
                split_data.sort(key=lambda item: (item[0], item[1]))
                output_filepath = os.path.join(output_dir, split_name, f"{scene}_{video}.txt")
                
                with open(output_filepath, 'w') as out_file:
                    for t in split_data:
                        out_file.write(f"{t[0]:.1f}\t{t[1]:.1f}\t{t[2]:.2f}\t{t[3]:.2f}\n")

    print("\nProcesso concluído! Os vídeos foram fatiados preservando a cronologia e a proporção 70/10/20.")

if __name__ == "__main__":
    raw_dataset_path = "./datasets/sdd_raw"
    final_dataset_path = "./datasets/SDD"
    
    preprocess_sdd(raw_dataset_path, final_dataset_path)