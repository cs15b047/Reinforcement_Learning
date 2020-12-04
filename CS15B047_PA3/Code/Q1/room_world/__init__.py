from gym.envs.registration import register

register(
    id='room_world-v0',
    entry_point='room_world.FourRooms:FourRooms',
)
