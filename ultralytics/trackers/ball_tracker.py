import os, sys, time
import numpy as np
from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker
from .utils import matching


class BallTracker(BYTETracker):
    def get_dists(self, tracks, detections):
        """Calculates distances between tracks and detections using IoU and optionally ReID embeddings."""

        dists = matching.pos_distance(tracks, detections, self.args.dist_ratio)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
            
        return dists


