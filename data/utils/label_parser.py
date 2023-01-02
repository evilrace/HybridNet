def parse_kitti_object(obj):
    cls_type = obj[0]
    left, top, right, bottom = list(map(float,obj[4:8]))
    x, y, w, h = convert_corners_to_xywh(left, top, right, bottom)
    return (cls_type, x, y, w, h)

def convert_corners_to_xywh(left, top, right, bottom):
    x = (left + right) / 2.0
    y = (top + bottom) / 2.0
    w = right - left
    h = bottom - top
    return x,y,w,h

