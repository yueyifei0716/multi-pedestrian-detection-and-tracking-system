# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track

# Tracker类是最核心的类，Tracker中保存了所有的轨迹信息
# 负责初始化第一帧的轨迹、卡尔曼滤波的预测和更新、负责级联匹配、IOU匹配

class Tracker:
    # 是一个多目标tracker，保存了很多个track轨迹
    # 负责调用卡尔曼滤波来预测track的新状态+进行匹配工作+初始化第一帧
    # Tracker调用update或predict的时候，其中的每个track也会各自调用自己的update或predict
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
        测量与轨迹关联的距离度量
    max_age : int
        Maximum number of missed misses before a track is deleted.
        删除轨迹前的最大未命中数
    n_init : int
        Number of frames that a track remains in initialization phase.
        确认轨迹前的连续检测次数。如果前n_init帧内发生未命中，则将轨迹状态设置为Deleted
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric  # metric是一个类，用于计算余弦距离或者马氏距离
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age  # 直接指定级联匹配的cascade_depth参数
        self.n_init = n_init  # 需要n_init次数的update才会将track状态设置为confirmed

        self.kf = kalman_filter.KalmanFilter()  # 实例化卡尔曼滤波器
        self.tracks = []   # 保存一个轨迹列表，用于保存一系列轨迹
        self._next_id = 1  # 下一个分配的轨迹id
 
    def predict(self):
        # 遍历每个track都用卡尔曼滤波算法进行预测
        """Propagate track state distributions one time step forward.
        将跟踪状态分布向前传播一步

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

# -------------------------update函数-------------------------------
    def update(self, detections):
        """Perform measurement update and track management.
        执行测量更新和轨迹管理

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        # 函数的入口，通过级联匹配得到匹配对、未匹配的tracks、未匹配的dectections
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)



        # Update track set.
        # 对于每个匹配成功的track，用其对应的detection进行更新
        # 1. 针对匹配上的结果
        for track_idx, detection_idx in matches:
            # 更新tracks中相应的detection
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        
        # 2. 针对未匹配的track, 调用mark_missed进行标记，将其标记为丢失
        # track失配时，若Tantative则删除；若update时间很久也删除
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # 3. 针对未匹配的detection， detection失配，进行初始化为新的track
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        
        # 得到最新的tracks列表，保存的是标记为Confirmed和Tentative的track
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            # 获取所有Confirmed状态的track id
            if not track.is_confirmed():
                continue
            features += track.features  # 将Confirmed状态的track的features添加到features列表
            # 获取每个feature对应的trackid
            targets += [track.track_id for _ in track.features]
            track.features = []
        # 距离度量中的特征集更新
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    # ------------------------------match函数-----------------------------------
    def _match(self, detections):
        # 主要功能是进行匹配，找到匹配的，未匹配的部分
        # 返回一个状态为confirmed的跟踪对象和检测对象的一个代价矩阵
        def gated_metric(tracks, dets, track_indices, detection_indices):
            # gated_metric计算track和detection之间的距离，返回一个代价矩阵
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            
            # 通过最近邻（余弦距离）计算出成本矩阵（代价矩阵）
            # 基于外观信息，计算tracks和detections的余弦距离代价矩阵
            cost_matrix = self.metric.distance(features, targets)
            # 计算门控后的成本矩阵（代价矩阵:每个track对象和现在det对象的代价值）
            # 使用马氏距离，大于阈值的都把代价变成无穷。
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # 区分开confirmed tracks和unconfirmed tracks
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # 对确定态的轨迹进行级联匹配，得到匹配的tracks、不匹配的tracks、不匹配的detections
        # matching_cascade级联匹配 根据特征将检测框匹配到确认的轨迹。
        # 传入门控后的成本矩阵
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.        
        # 将未确定态的轨迹和刚刚没有匹配上的轨迹组合为 iou_track_candidates 
        # 并进行基于IoU的匹配
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]  # 刚刚没有匹配上的轨迹
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]  # 并非刚刚没有匹配上的轨迹

        # 对级联匹配中还没有匹配成功的目标再进行IoU匹配
        # min_cost_matching 使用匈牙利算法解决线性分配问题。
        # 传入 iou_cost，尝试关联剩余的轨迹与未确认的轨迹。
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        # 整合所有的匹配对和未匹配的tracks
        matches = matches_a + matches_b  # 组合两部分匹配
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections


    def _initiate_track(self, detection):
        #初始化新的track
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
