package com.example.client_trial2.Networking.Helpers;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class NetHelper
{
    public static byte[] AppendHeader(byte[] Message, int HeaderSize)
    {
        byte[] Packet = new byte[Message.length + HeaderSize];
        byte[] CodeBytes = ByteBuffer.allocate(HeaderSize).order(ByteOrder.LITTLE_ENDIAN).putInt(Message.length).array();
        System.arraycopy(CodeBytes, 0, Packet, 0, HeaderSize);
        System.arraycopy(Message, 0, Packet, HeaderSize, Message.length);
        return Packet;
    }

    public static int GetInt(byte[] Bytes)
    {
        ByteBuffer wrapped = ByteBuffer.wrap(Bytes);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        return wrapped.getInt();
    }
}
