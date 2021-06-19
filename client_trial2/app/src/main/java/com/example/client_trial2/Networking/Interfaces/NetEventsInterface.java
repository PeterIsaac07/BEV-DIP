package com.example.client_trial2.Networking.Interfaces;


import com.example.client_trial2.Networking.ClientSocket;
import com.example.client_trial2.Networking.Models.Packet;

public interface NetEventsInterface
{
    void SetOnClientConnect(ClientSocket Client);
    void SetOnClientSend(ClientSocket Client, Packet Message);
    void SetOnClientReceive(ClientSocket Client, byte[] Message);
    void SetOnClientDisconnect(ClientSocket Client);
    void SetOnClientException(Exception Ex);
}