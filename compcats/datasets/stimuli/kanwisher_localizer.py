"""
Code to load Kanwisher localizer
Julian, J. B., Fedorenko, E., Webster, J., & Kanwisher, N. (2012). An algorithmic method for functionally defining regions of interest in the ventral visual pathway. Neuroimage, 60(4), 2357-2364.
"""
import glob

from .base_localizer import BaseLocalizerDataset

class KanwisherLocalizerDataset(BaseLocalizerDataset):
    """
    Dataset for Kanwisher localizer
    """
    def __init__(self, root_dir, transform, image_class='all'):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        image_class = image_class.strip().lower()
        self.image_class = image_class
        self.name = f'kanwisher_{image_class}'
        if image_class == 'allbodies':
            self.image_files = sorted(glob.glob(f'{root_dir}/Bodies/*.png'))
        elif image_class == 'bodies':
            self.image_files = glob.glob(f'{root_dir}/Bodies/*.png')
        elif image_class == 'allfaces':
            self.image_files = sorted(glob.glob(f'{root_dir}/Faces/*.png'))
        elif image_class == 'faces':
            self.image_files = glob.glob(f'{root_dir}/Faces/*.png')
        elif image_class == 'allobjects':
            self.image_files = sorted(glob.glob(f'{root_dir}/Objects/*.png'))
        elif image_class == 'objects':
            self.image_files = glob.glob(f'{root_dir}/Objects/*.png')
        elif image_class == 'allscenes':
            self.image_files = sorted(glob.glob(f'{root_dir}/Scenes/*.png'))
        elif image_class == 'scenes':
            self.image_files = glob.glob(f'{root_dir}/Scenes/*.png')
        elif image_class == 'allscrambledobjects':
            self.image_files = sorted(glob.glob(f'{root_dir}/scrambled15G/*.png'))
        elif image_class == 'scrambledobjects':
            self.image_files = glob.glob(f'{root_dir}/scrambled15G/*.png')
        elif image_class == 'all':
            self.image_files = sorted(glob.glob(f'{root_dir}/**/*.png'))
        else:
            raise ValueError(f'{image_class} images not implemented in localizer')
        # Contrasts
        self.froi_contrasts = {
            'FFA': {
                'preffered': 'faces',
                'nonpreffered': ['bodies', 'scenes', 'objects', 'scrambledobjects']
            },
            'PPA': {
                'preffered': 'scenes',
                'nonpreffered': ['bodies', 'faces', 'objects', 'scrambledobjects']
            },
            'EBA': {
                'preffered': 'bodies',
                'nonpreffered': ['faces', 'scenes', 'objects', 'scrambledobjects']
            },
        }
        self.imageclasses = [
            'faces', 'bodies', 'scenes', 'objects', 'scrambledobjects'
        ]
        # Number of samples per category
        # 288 for bodies, faces, scenes, objects
        # 144 for scrambledobjects
        # Matched to VPNL-fLoc localizer
        # https://github.com/VPNL/fLoc
        self.samples_per_category = {
            'bodies': 288,
            'faces': 288,
            'scenes': 288,
            'objects': 288,
            'scrambledobjects': 144
        }
        # Sample images
        if image_class in ['bodies', 'faces', 'objects', 'scenes', 'scrambledobjects']:
            self.image_files = self._sample_images()
    
    def _sample_images(self):
        """
        Function to subsample images to match VPNL-fLoc localizer
        For each video, take the first frame and the middle frame, then take the frame at 
        1/4 of the video until number of samples is reached
        288 for bodies, faces, scenes, objects
        144 for scrambledobjects
        """
        image_files = self.image_files
        # Sample images list
        sampled_image_files = []
        # Useful variables
        base_path = '/'.join(image_files[0].split('/')[:-1])
        if self.image_class == 'scrambledobjects':
            keyword = image_files[0].split('/')[-1].split('_')[1]
        else:
            keyword = image_files[0].split('/')[-1].split('_')[0]
        num_samples = self.samples_per_category[self.image_class]
        # Get image files for each video
        video_frames = {}
        for image_file in image_files:
            parts = image_file.split('/')[-1].split('_')
            if self.image_class == 'scrambledobjects':
                video_number = parts[0]
            else:
                video_number = parts[1]
            if video_number not in video_frames:
                video_frames[video_number] = []
            video_frames[video_number].append(image_file)
        # Sample images
        num_videos = len(video_frames)
        # Error handling
        assert num_videos == 60, f'Expected 60 videos but found {num_videos}'
        num_frames_from_all_videos = num_samples // num_videos
        assert num_frames_from_all_videos in [2, 4], \
            f'Expected 2 or 4 frames from each video but found {num_frames_from_all_videos}'
        for video_number, frames in video_frames.items():
            # Get number of frames
            num_frames = len(frames)
            if self.image_class == 'scrambledobjects':
                # Append first frame
                sampled_image_files.append(
                    f'{base_path}/{video_number}_{keyword}_f0.png'
                )
                # Append middle frame
                sampled_image_files.append(
                    f'{base_path}/{video_number}_{keyword}_f{num_frames//2}.png'
                )
            else:
                # Append first frame
                sampled_image_files.append(
                    f'{base_path}/{keyword}_{video_number}_f0.png'
                )
                # Append middle frame
                sampled_image_files.append(
                    f'{base_path}/{keyword}_{video_number}_f{num_frames//2}.png'
                )
                # Append 1/4 frame
                sampled_image_files.append(
                    f'{base_path}/{keyword}_{video_number}_f{num_frames//4}.png'
                )
                # Append 3/4 frame
                sampled_image_files.append(
                    f'{base_path}/{keyword}_{video_number}_f{3*num_frames//4}.png'
                )
        # If num_samples is not reached then take 1/8 frames from each video
        # until num_samples is reached
        samples_remaining = num_samples - len(sampled_image_files)
        if samples_remaining > 0:
            for video_number, frames in video_frames.items():
                num_frames = len(frames)
                if self.image_class == 'scrambledobjects':
                    # Append 1/4 frame
                    sampled_image_files.append(
                        f'{base_path}/{video_number}_{keyword}_f{num_frames//4}.png'
                    )
                else:
                    # Append 1/8 frame
                    sampled_image_files.append(
                        f'{base_path}/{keyword}_{video_number}_f{num_frames//8}.png'
                    )
                samples_remaining -= 1
                if samples_remaining == 0:
                    break
        # Error handling
        if samples_remaining > 0:
            raise ValueError(f'Could not sample {samples_remaining} images')
        # Sort sampled images
        sampled_image_files = sorted(sampled_image_files)
        # Return sampled image files
        return sampled_image_files


def get_localizer_dataset(root_dir, transform, image_class='all'):
    """
    Get Kanwisher Localizer dataset
    Args:
        root_dir(str): Root directory
        transform(torchvision.transforms): Image transformations
    """
    dataset = KanwisherLocalizerDataset(root_dir, transform=transform, image_class=image_class)
    return dataset
