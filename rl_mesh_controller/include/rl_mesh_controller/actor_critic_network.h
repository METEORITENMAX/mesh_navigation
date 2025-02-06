#ifndef RL_MESH_CONTROLLER__ACTOR_CRITIC_NETWORK_H
#define RL_MESH_CONTROLLER__ACTOR_CRITIC_NETWORK_H

#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <pwd.h>
#include <iostream>
#include <cstring>
#include <cerrno>

class ActorCriticNetwork {
public:
    ActorCriticNetwork();
    ~ActorCriticNetwork();

    void initializeGraph();

    void SaveModel();

private:


};

#endif  // RL_MESH_CONTROLLER__ACTOR_CRITIC_NETWORK_H
