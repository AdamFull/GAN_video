import cv2, os, hashlib
import config
import discord
from discord.ext import commands, tasks
from discord.utils import get
from itertools import cycle
from random import seed, random, choice

from gan import GAN

seed(10)
self_path = os.path.dirname(os.path.abspath(__file__))
temp_path = os.path.join(self_path, "temp")

if not os.path.exists(temp_path):
    os.makedirs(temp_path)

generator = GAN(buff_size=10035, batch_size=16, epochs=5000, imgs_size=(config.width, config.heigth))

def get_image(user_uid):
    image = generator.generate_image()
    img_name = os.path.join(temp_path, hashlib.sha256((str(user_uid)+str(random.random())).encode()).hexdigest() + ".jpg")
    cv2.imwrite(img_name, image)
    return img_name


#---------------------DISCORD---------------------
client = commands.Bot(command_prefix= '!')
statuses = cycle(["Playing Dota", "Fapping", "Watching porn",
                 "Having sex with Cedric", "Watching GacmiMuchi", 
                 "Doing hard job", "Fisting"])

@client.event
async def on_ready():
    await client.change_presence(status=discord.Status.idle, activity=discord.Game('PornHub'))
    change_status.start()
    print('Logged on as {0}'.format(client.user))

@client.event
async def on_member_join(member):
    role = discord.utils.get(member.server.roles, name="@user")
    await client.add_roles(member, role)

@client.event
async def on_member_remove(member):
    pass

@client.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Required another argument type.")
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Invalid command, use !help.")

@client.command()
async def harder_daddy(ctx):
    img_name = get_image(ctx.author.id)
    await ctx.send('Wait, loading')
    await ctx.send(file=discord.File(img_name))
    os.remove(img_name)

@client.command()
async def ping(ctx):
    await ctx.send(f'Suck: {round(client.latency * 1000)} ms')

@client.command()
@commands.has_permissions(manage_messages=True)
async def clear(ctx, amount : int):
    await ctx.channel.purge(limit=amount)

@commands.has_permissions(kick_members=True)
@client.command()
async def kick(ctx, member : discord.Member, *, reason=None):
    await member.kick(reason=reason)
    await ctx.send(f"Kicked {member.mention}")

@commands.has_permissions(ban_members=True)
@client.command()
async def ban(ctx, member : discord.Member, *, reason=None):
    await member.ban(reason=reason)
    await ctx.send(f"Banned {member.mention}")

@commands.has_permissions(ban_members=True)
@client.command()
async def unban(ctx, *, member):
    banned = await ctx.guild.bans()
    member_name, member_discriminator = member.split('#')

    for ban_entry in banned:
        user = ban_entry.user
        if (user.name, user.discriminator) == (member_name, member_discriminator):
            await ctx.guild.unban(user)
            await ctx.send(f"Unbanned {user.mention}")
            return

@client.command()
async def load(ctx, extention):
    client.load_extension(f"cogs.{extention}")

@client.command()
async def unload(ctx, extention):
    client.unload_extension(f"cogs.{extention}")

@client.command()
async def noice(ctx):
    await ctx.send("https://tenor.com/view/noice-nice-click-gif-8843762")
    await ctx.send("clock, noice", tts=True)

@client.command()
async def who_is_gay_today(ctx):
    members = ctx.guild.members
    gay = choice(members)
    await ctx.send(f"Today gay is {gay.mention}")

@client.command()
async def friend_me(ctx):
    idf = ctx.author.id
    user = await client.get_user(idf)
    await user.send_friend_request()
    

@client.command()
async def wise(ctx):
    await ctx.send("Cedric is a gay i guess...",tts=True)

@tasks.loop(seconds=60)
async def change_status():
    await client.change_presence(activity=discord.Game(next(statuses)))

client.run(config.DC_TOKEN)