# ROAR_Competition

The first place submission by the BHS Racing team for the Summer 2024 ROAR Competition on the Monza map.

## Requirements

Requires `progress` module and ROAR_PY

## Time spread

Official Run:
325.6 seconds

Unofficial time over 10 runs:

- Average: 325.55 seconds
- Fastest: 325.250 seconds
- Slowest: 325.850 seconds

Reliability:

- 10 runs with no crashes
- 20 runs with 2 crashes

> [!NOTE]
> Run-to-run variance can cause crashes or cause a slower or faster time. Repeated testing with debugCompetitionRunner.py is recommended to filter out outliers.

## Waypoints

To create new waypoints, you have to run the waypoint_collect.py file and select your starting location, and then drive around the track until you have collected all your desired waypoints. If you only want to do a section, close the Pygame window and the program will save the waypoints you drove.

Then, you can either use these new waypoints as is or splice them into an existing waypoint file with waypointSplicer.py. Running waypointSplicer.py will display a graph of the base waypoint file (the file with the primary waypoints) and the new waypoints you want to substitute in. Click on the end points of the section you want to replace, or double click if you want to replace from that point to the end of the track.

Using the new waypoints can be done in the `initialize` function of `ROARCompetitionSolution`.

Collecting the track's points can be done with `collectBaseWaypoints.py`.
